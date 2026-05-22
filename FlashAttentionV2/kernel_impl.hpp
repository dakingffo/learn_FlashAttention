#include <cute/tensor.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cutlass/numeric_conversion.h>

#include <utility/math.hpp>
#include <utility/params.hpp>

namespace FlashAttention::V2 {
    template <
        size_t Dim, size_t CTATile, bool FillOutOfBoundary = true,
        typename TiledCopy, typename Identity,
        typename SrcTensor, typename DstTensor
    >
    CUTE_DEVICE void copy_within_boundary(
        TiledCopy&& tiled_copy, 
        signed long len, unsigned int idx, 
        const Identity& identity,
        const SrcTensor& src, DstTensor&& dst,
        typename std::decay_t<DstTensor>::value_type val = typename std::decay_t<DstTensor>::value_type{0.0}
    ) {
        if ((idx + 1) * CTATile - 1 < len) {
            copy(tiled_copy, src, dst);
        }
        else {
            auto mask = make_tensor<bool>(shape(identity));
            CUTE_UNROLL
            for (int i = 0; i < size(identity); i++) {
                bool within_boundary = (idx * CTATile + get<Dim>(identity(i)) < len);
                mask(i) = within_boundary;
                if constexpr (FillOutOfBoundary) {
                    dst(i) = (within_boundary ? dst(i) : val);
                }
            }
            copy_if(tiled_copy, mask, src, dst);
        }
    }

    template <size_t Dim, size_t CTATile, typename Tensor, typename Identity>
    CUTE_DEVICE void fill_cross_boundary(
        Tensor&& tensor, 
        signed long len, unsigned int idx,
        const Identity& identity,
        typename decay_t<Tensor>::value_type internal,
        typename decay_t<Tensor>::value_type external
    ) {
        if ((idx + 1) * CTATile - 1 < len) {
            fill(tensor, internal);
        }
        else {
            CUTE_UNROLL
            for (int i = 0; i < size(identity); i++) {
                tensor(i) = (idx * CTATile + get<Dim>(identity(i)) < len ? internal : external);
            }
        }
    }

    template <typename Traits, size_t HeadDim>
    __global__ __launch_bounds__(Traits::ThreadsPerCTA)
    void forward_kernel(
        __grid_constant__ const utility::Params params,
        typename Traits::const_pointer          q,
        typename Traits::const_pointer          k,
        typename Traits::const_pointer          v,
        typename Traits::pointer                out,
        typename Traits::lse_pointer            lse
    ) {        
        using type      = typename Traits::type;
        using acc_type  = typename Traits::acc_type;
        using lse_type  = typename Traits::lse_type;
        using pointer   = typename Traits::pointer;
        using MMA       = typename Traits::MMA;
        using G2SCopy   = typename Traits::G2SCopy;
        using S2RCopyA  = typename Traits::S2RCopyA;
        using S2RCopyB  = typename Traits::S2RCopyB;
        using S2RCopyBT = typename Traits::S2RCopyBT;
        using R2SCopyC  = typename Traits::R2SCopyC;
        using S2GCopy   = typename Traits::S2GCopy;
        using CLayout   = typename Traits::mma_traits::CLayout;

        static constexpr int TileQ          = Traits::TileQ;
        static constexpr int TileKV         = Traits::TileKV;
        static constexpr int Stage          = Traits::Stage;
        static constexpr int AtomM          = Traits::AtomM;
        static constexpr int PM             = Traits::PM;
        static constexpr int ThreadsPerRow  = size<0, 0>(CLayout{});
        static constexpr int RowsPerThread  = size<1, 1>(CLayout{}) * TileQ / Traits::PM;
        static constexpr int ThreadsPerCol  = size<0, 1>(CLayout{});
        static constexpr int ThreadsPerAtom = ThreadsPerRow * ThreadsPerCol;
        static constexpr int SwizzleBits    = 3;
        static constexpr int SwizzleBase    = utility::log2<16 / sizeof(type)>();
        static constexpr int SwizzleShift   = 3;
        static constexpr int BlockKSmem     = HeadDim % 64 == 0 ? 64 : 32;

        using SmemLayoutAtom     = decltype(
            make_layout(Shape<_8, Int<BlockKSmem>>{}, Stride<Int<BlockKSmem>, _1>{})
        );
        using SmemLayoutQLogical = decltype(tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<TileQ>{}, Int<HeadDim>{})
        ));
        using SmemLayoutKLogical = decltype(tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<TileKV>{}, Int<HeadDim>{}, Int<Stage>{})
        ));
        using SmemLayoutVLogical = decltype(tile_to_shape(
            SmemLayoutAtom{},
            make_shape(Int<TileKV>{}, Int<HeadDim>{}, Int<Stage>{})
        ));
        using SmemLayoutVTLogical   = decltype(select<1, 0, 2>(SmemLayoutVLogical{}));
        using SwizzleFn             = Swizzle<SwizzleBits, SwizzleBase, SwizzleShift>;
        using SmemLayoutQ           = decltype(composition(SwizzleFn{}, SmemLayoutQLogical{}));
        using SmemLayoutK           = decltype(composition(SwizzleFn{}, SmemLayoutKLogical{}));
        using SmemLayoutV           = decltype(composition(SwizzleFn{}, SmemLayoutVLogical{}));
        using SmemLayoutO           = SmemLayoutQ;
        using SmemLayoutVT          = decltype(composition(SwizzleFn{}, SmemLayoutVTLogical{}));
        using SmemLayoutVNoSwizzle  = SmemLayoutVLogical;
        using SmemLayoutVTNoSwizzle = SmemLayoutVTLogical;
        
        extern __shared__ char shared_mem[];

        Tensor sQ = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutQ{});
        // (TileQ, HeadDim)
        Tensor sK = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ{})), SmemLayoutK{});
        // (TileK, HeadDim, Stage)
        Tensor sV = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{})), SmemLayoutV{});
        // (TileV, HeadDim, Stage)
        Tensor sVt = make_tensor(sV.data(), SmemLayoutVT{}); 
        // (HeadDim, TileKV, Stage)
        Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVTNoSwizzle{}); 
        // (HeadDim, TileKV, Stage)

        auto mx  = make_tensor<acc_type>(Shape<Int<RowsPerThread>, _2>{}); // (RowsPerThread, 2) for old_max, new_max
        auto den = make_tensor<acc_type>(Shape<Int<RowsPerThread>>{});     // (RowsPerThread)  

        Layout GlobalLayout = make_layout(
            make_shape(params.batch_size, params.num_heads, params.seq_len, Int<HeadDim>{}),
            make_stride(params.one_batch_size(), params.one_head_size(), Int<HeadDim>{}, _1{})
        );
        Tensor Q = make_tensor(make_gmem_ptr(q), GlobalLayout)(blockIdx.x, blockIdx.y, _, _);   // (seq_len, HeadDim)
        Tensor K = make_tensor(make_gmem_ptr(k), GlobalLayout)(blockIdx.x, blockIdx.y, _, _);   // (seq_len, HeadDim)
        Tensor V = make_tensor(make_gmem_ptr(v), GlobalLayout)(blockIdx.x, blockIdx.y, _, _);   // (seq_len, HeadDim)
        Tensor O = make_tensor(make_gmem_ptr(out), GlobalLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)

        /*
            For batch = blockIdx.x and head = blockIdx.y:
            Each cta will compute a TileQ x head_dim tile of Q, then loop over seq_len / TileKV tiles of K and V
            [    Q0    ] <- blockIdx.z = 0  [    ][    ][    ]                    [    V0    ]
            [    Q1    ]         *          [ K0 ][ K1 ][ K2 ]    -SoftMax-> *    [    V1    ]
            [    Q2    ]                    [    ][    ][    ]                    [    V2    ]
        */

        Tensor gQ = local_tile(Q, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileQ, HeadDim)
        Tensor gK = local_tile(K, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(_, 0));         // (TileK, HeadDim, num_tiles_k)
        Tensor gV = local_tile(V, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(_, 0));         // (TileV, HeadDim, num_tiles_v)
        Tensor gO = local_tile(O, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileQ, HeadDim)

        auto iQ = make_identity_tensor(shape(gQ));
        auto iK = make_identity_tensor(shape(gK(_, _, 0)));
        auto iV = make_identity_tensor(shape(gV(_, _, 0)));
        auto iS = make_identity_tensor(Shape<Int<TileQ>, Int<TileKV>>{});
        auto iO = make_identity_tensor(shape(gO));

        MMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
        Tensor tSrQ = thr_mma.partition_fragment_A(gQ);                                  // (MMA, MMA_TileQ, MMA_HeadDim)
        Tensor tSrK = thr_mma.partition_fragment_B(gK(_, _, 0));                         // (MMA, MMA_TileK, MMA_HeadDim)
        Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<TileQ>, Int<TileKV>>{}); // (MMA, MMA_TileQ, MMA_TileK)
        // tSrS clear in loop

        auto tOrP  = thr_mma.partition_fragment_A(make_tensor<type>(Shape<Int<TileQ>, Int<TileKV>>{}));   
                                                                            // (MMA, MMA_TileQ, MMA_TileK)
        Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, 0)); // (MMA, MMA_HeadDim, MMA_TileV)
        Tensor tOrO  = thr_mma.partition_fragment_C(gO);                    // (MMA, MMA_TileQ, MMA_HeadDim)
        clear(tOrO);

        // if (cute::thread0()) {
        //     print(CLayout{}); printf("\n");
        //     print(tiled_mma.get_layoutC_TV()); printf("\n");
        //     print(tiled_mma.get_layoutC_TV()); printf("\n");
        //     print(tiled_mma.get_layoutC_TV()); printf("\n");
        //     printf("\n");
        //     print(tSrQ.layout()); printf("\n");
        //     print(tSrK.layout()); printf("\n");
        //     print(tSrS.layout()); printf("\n");
        //     printf("\n");
        //     print(tOrP.layout()); printf("\n");
        //     print(tOrVt.layout()); printf("\n");
        //     print(tOrO.layout()); printf("\n");
        // }
        // __syncthreads();

        G2SCopy g2s_copy_q, g2s_copy_k, g2s_copy_v;
        ThrCopy thr_g2s_copy_q = g2s_copy_q.get_slice(threadIdx.x);
        Tensor tSgQ_g2s_view = thr_g2s_copy_q.partition_S(gQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor tSsQ_g2s_view = thr_g2s_copy_q.partition_D(sQ); // (COPY, COPY_TileQ, COPY_HeadDim)

        ThrCopy thr_g2s_copy_k = g2s_copy_k.get_slice(threadIdx.x);
        Tensor tSgK_g2s_view = thr_g2s_copy_k.partition_S(gK); // (COPY, COPY_TileK, COPY_HeadDim, num_tiles_k)
        Tensor tSsK_g2s_view = thr_g2s_copy_k.partition_D(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)

        ThrCopy thr_g2s_copy_v = g2s_copy_v.get_slice(threadIdx.x);
        Tensor tOgV_g2s_view = thr_g2s_copy_v.partition_S(gV); // (COPY, COPY_TileV, COPY_HeadDim, num_tiles_v)
        Tensor tOsV_g2s_view = thr_g2s_copy_v.partition_D(sV); // (COPY, COPY_TileV, COPY_HeadDim, Stage)

        S2RCopyA s2r_copy_q;
        ThrCopy thr_s2r_copy_q = s2r_copy_q.get_slice(threadIdx.x);
        Tensor tSsQ_s2r_view = thr_s2r_copy_q.partition_S(sQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor tSrQ_s2r_view = thr_s2r_copy_q.retile_D(tSrQ);  // (COPY, COPY_TileQ, COPY_HeadDim)
        
        S2RCopyB s2r_copy_k;
        ThrCopy thr_s2r_copy_k = s2r_copy_k.get_slice(threadIdx.x);
        Tensor tSsK_s2r_view = thr_s2r_copy_k.partition_S(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)
        Tensor tSrK_s2r_view = thr_s2r_copy_k.retile_D(tSrK);  // (COPY, COPY_TileK, COPY_HeadDim)
        
        S2RCopyBT s2r_copy_v;
        ThrCopy thr_s2r_copy_v = s2r_copy_v.get_slice(threadIdx.x);
        Tensor tOsVt_s2r_view = thr_s2r_copy_v.partition_S(sVt); // (COPY, COPY_HeadDim, COPY_TileV, Stage)
        Tensor tOrVt_s2r_view = thr_s2r_copy_v.retile_D(tOrVt);  // (COPY, COPY_HeadDim, COPY_TileV)

        int global_read = 0, smem_pipe_read = 0, smem_pipe_write = 0;

        // copy(g2s_copy_q, tSgQ_g2s_view, tSsQ_g2s_view);
        copy_within_boundary<0, TileQ>(
            g2s_copy_q, params.seq_len, blockIdx.z, 
            thr_g2s_copy_q.partition_S(iQ),
            tSgQ_g2s_view, tSsQ_g2s_view
        );
        cp_async_fence();
        CUTE_UNROLL
        for (; smem_pipe_write < min(Stage - 1, ceil_div(params.seq_len, TileKV)); global_read++, smem_pipe_write++) {
            copy_within_boundary<0, TileKV>(
                g2s_copy_k, params.seq_len, global_read, 
                thr_g2s_copy_k.partition_S(iK),
                tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
            );
            copy_within_boundary<0, TileKV>(
                g2s_copy_v, params.seq_len, global_read, 
                thr_g2s_copy_v.partition_S(iV),
                tOgV_g2s_view(_, _, _, global_read), tOsV_g2s_view(_, _, _, smem_pipe_write)
            );
            cp_async_fence();
        }
        cp_async_wait<Stage - 2>(); // Q is ready
        __syncthreads();

        int mx_idx = 0; 
        CUTE_UNROLL
        for (int kv_idx = 0; kv_idx < ceil_div(params.seq_len, TileKV); kv_idx++, mx_idx ^= 1) {
            // clear(tSrS);
            fill_cross_boundary<1, TileKV>(
                tSrS, params.seq_len, kv_idx, 
                thr_mma.partition_fragment_C(iS),
                acc_type{0.0}, numeric_limits<acc_type>::lowest()
            );
            // Q * K -> S
            if (kv_idx == 0) {
                copy(s2r_copy_q, tSsQ_s2r_view(_, _, 0), tSrQ_s2r_view(_, _, 0));
            }
            copy(s2r_copy_k, tSsK_s2r_view(_, _, 0, smem_pipe_read), tSrK_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrQ); i++) {
                if (i == 0) {
                    if (global_read < ceil_div(params.seq_len, TileKV)) {
                        copy_within_boundary<0, TileKV>(
                            g2s_copy_k, params.seq_len, global_read, 
                            thr_g2s_copy_k.partition_S(iK),
                            tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
                        );
                        copy_within_boundary<0, TileKV>(
                            g2s_copy_v, params.seq_len, global_read, 
                            thr_g2s_copy_v.partition_S(iV),
                            tOgV_g2s_view(_, _, _, global_read), tOsV_g2s_view(_, _, _, smem_pipe_write)
                        );
                        global_read++;
                        smem_pipe_write = (smem_pipe_write + 1) % Stage;
                    }
                    cp_async_fence();
                }
                if (i + 1 < size<2>(tSrQ)) {
                    copy(s2r_copy_q, tSsQ_s2r_view(_, _, i + 1), tSrQ_s2r_view(_, _, i + 1));
                    copy(s2r_copy_k, tSsK_s2r_view(_, _, i + 1, smem_pipe_read), tSrK_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tSrS, tSrQ(_, _, i), tSrK(_, _, i), tSrS);
            }
            
            // Online SoftMax(S) -> P
            CUTE_UNROLL
            for (int i = 0; i < RowsPerThread; i++) {
                int a = i % size<0, 1>(tSrS), b = i / size<0, 1>(tSrS);
                acc_type new_max = (kv_idx != 0 ? mx(i, mx_idx) : numeric_limits<acc_type>::lowest());
                CUTE_UNROLL
                for (int k = 0; k < size<2>(tSrS); k++)  {
                    CUTE_UNROLL
                    for (int j = 0; j < size<0, 0>(tSrS); j++) {
                        new_max = max(new_max, tSrS(make_coord(j, a), b, k));
                    }
                }
                CUTE_UNROLL
                for (int offset = ThreadsPerRow >> 1; offset; offset >>= 1) {
                    new_max = max(new_max, __shfl_xor_sync(0xffffffff, new_max, offset));
                }
                mx(i, mx_idx ^ 1) = new_max;

                acc_type delta_den = 0.0;
                cute::transform(
                    tSrS(make_coord(_, a), b, _),
                    [new_max, &delta_den, scale = params.scale] __device__ (acc_type ele) { 
                        acc_type exp_ele = expf((ele - new_max) * scale);
                        delta_den += exp_ele;
                        return exp_ele;
                    }
                );
                CUTE_UNROLL
                for (int offset = ThreadsPerRow >> 1; offset; offset >>= 1) {
                    delta_den += __shfl_xor_sync(0xffffffff, delta_den, offset);
                }
                if (kv_idx != 0) {
                    acc_type rescale = expf((mx(i, mx_idx) - new_max) * params.scale);
                    cute::transform(
                        tOrO(make_coord(_, a), b, _),
                        [rescale] __device__ (acc_type ele) {
                            return ele * rescale;
                        }
                    );
                    den(i) = den(i) * rescale + delta_den;
                }
                else {
                    den(i) = delta_den;
                }
            }
            
            // P * V -> O
            copy(tSrS, tOrP);
            copy(s2r_copy_v, tOsVt_s2r_view(_, _, 0, smem_pipe_read), tOrVt_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tOrP); i++) {
                if (i + 1 < size<2>(tOrP)) {
                    copy(s2r_copy_v, tOsVt_s2r_view(_, _, i + 1, smem_pipe_read), tOrVt_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tOrO, tOrP(_, _, i), tOrVt(_, _, i), tOrO);
            }
            cp_async_wait<Stage - 2>();
            __syncthreads();
            smem_pipe_read = (smem_pipe_read + 1) % Stage;
        }
        
        CUTE_UNROLL
        for (int i = 0; i < RowsPerThread; i++) {
            int a = i % size<0, 1>(tOrO), b = i / size<0, 1>(tOrO);
            cute::transform(
                tOrO(make_coord(_, a), b, _),
                [den = den(i)] __device__ (acc_type ele) {
                    return ele / den;
                }
            );
        }

        Tensor rO = make_tensor(tSrQ.data(), tOrO.layout());                        // acc_type -> type
        Tensor sO = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutO{}); // (TileQ, HeadDim)
        copy(tOrO, rO);
        R2SCopyC r2s_copy_o;
        ThrCopy thr_r2s_copy_o = r2s_copy_o.get_slice(threadIdx.x);
        Tensor rO_r2s_view = thr_r2s_copy_o.retile_S(rO);    // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor sO_r2s_view = thr_r2s_copy_o.partition_D(sO); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads(); // make sure nobody using sO
        copy(r2s_copy_o, rO_r2s_view, sO_r2s_view);
        S2GCopy s2g_copy_o;
        ThrCopy thr_s2g_copy_o = s2g_copy_o.get_slice(threadIdx.x);
        Tensor sO_s2g_view = thr_s2g_copy_o.partition_S(sO); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor gO_s2g_view = thr_s2g_copy_o.partition_D(gO); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads();
        copy_within_boundary<0, TileQ, false>(
            s2g_copy_o, params.seq_len, blockIdx.z, 
            thr_s2g_copy_o.partition_D(iO),
            sO_s2g_view, gO_s2g_view
        );

        auto LSELayout = make_layout(
            make_shape(params.batch_size, params.num_heads, params.seq_len),
            make_stride(params.num_heads * params.seq_len, params.seq_len, _1{})
        );
        Tensor LSE  = make_tensor(make_gmem_ptr(lse), LSELayout)(blockIdx.x, blockIdx.y, _);
        Tensor gLSE = local_tile(LSE, Tile<Int<TileQ>>{}, make_coord(blockIdx.z));
        if (threadIdx.x % ThreadsPerRow == 0) {
            for (int i = 0; i < RowsPerThread; i++) {
                int a = i % size<0, 1>(tOrO), b = i / size<0, 1>(tOrO);
                int idx = b * PM + threadIdx.x / ThreadsPerAtom * AtomM + 
                          a * ThreadsPerCol + threadIdx.x % ThreadsPerAtom / ThreadsPerRow; 
                if (blockIdx.z * TileQ + idx < params.seq_len) {
                    gLSE(idx) = (float)mx(i, mx_idx) * params.scale + logf((float)den(i));
                }
            }
        }
    }
}