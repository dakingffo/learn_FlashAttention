#include <cute/tensor.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cutlass/numeric_conversion.h>
#include <iostream>

#include <utility/math.hpp>
#include <utility/parameters.hpp>
#include <utility/torch_utils.hpp>
#include <utility/type_traits.hpp>
#include <tuple>

namespace FlashAttention::V2 {
    using namespace cute;
    using namespace cutlass;

    template <
        size_t   QPerCTA  = 128,
        size_t   TileHead = 64,
        size_t   EUReapt  = 8,
        size_t   Pipe     = 5,
        typename MMAOP    = SM80_16x8x16_F32F16F16F32_TN
    >
    struct Traits {
        static constexpr int TileQ   = QPerCTA;
        static constexpr int TileKV  = TileHead;
        static constexpr int Stage   = Pipe;

        using mma_op         = MMAOP;
        using mma_traits     = MMA_Traits<mma_op>;
        using mma_atom       = MMA_Atom<mma_traits>;
        using mma_atom_shape = typename mma_traits::Shape_MNK;

        static_assert(std::is_same_v<typename mma_traits::ValTypeA, typename mma_traits::ValTypeB>);
        static_assert(std::is_same_v<typename mma_traits::ValTypeD, typename mma_traits::ValTypeC>);
        using type          = typename mma_traits::ValTypeA;
        using acc_type      = typename mma_traits::ValTypeD;
        using torch_type    = utility::torch_type_of_t<type>;
        using pointer       = type*;
        using const_pointer = const type*;

        static constexpr int EURepeatM = EUReapt;
        static constexpr int PermuteN  = get<2>(mma_atom_shape{}) / get<1>(mma_atom_shape{});
        static constexpr int PM        = EURepeatM * get<0>(mma_atom_shape{});
        static constexpr int PN        = PermuteN * get<1>(mma_atom_shape{});
        static constexpr int PK        = get<2>(mma_atom_shape{}); // only one warp
        static_assert(TileQ >= PM && TileKV >= PN);

        using MMA = decltype(make_tiled_mma(
            mma_op{},
            make_layout(Shape<Int<EURepeatM>, _1, _1>{}),
            Tile<Int<PM>, Int<PN>, Int<PK>>{}
        ));

        static constexpr int NumThreads = thr_size(MMA{});
        using CopyThreadsLayout = decltype(make_layout(Shape<Int<NumThreads / 4>, _4>{}, Stride<_4, _1>{}));

        using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, type>;
        using G2SCopy        = decltype(make_tiled_copy(
            g2s_copy_atom{},
            CopyThreadsLayout{},
            make_layout(Shape<_1, _8>{})
        ));

        using s2r_copy_op     = SM75_U32x4_LDSM_N;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, type>;
        using S2RCopyA        = decltype(make_tiled_copy_A(
            s2r_copy_atom{},
            MMA{}
        ));
        using S2RCopyB        = decltype(make_tiled_copy_B(
            s2r_copy_atom{},
            MMA{}
        ));

        using s2r_copy_op_T     = SM75_U16x8_LDSM_T;
        using s2r_copy_traits_T = Copy_Traits<s2r_copy_op_T>;
        using s2r_copy_atom_T   = Copy_Atom<s2r_copy_traits_T, type>;
        using S2RCopyBT         = decltype(make_tiled_copy_B(
            s2r_copy_atom_T{},
            MMA{}
        ));

        using r2s_copy_atom   = Copy_Atom<UniversalCopy<int>, type>;
        using R2SCopyC        = decltype(make_tiled_copy_C(
            r2s_copy_atom{},
            MMA{}
        ));

        using s2g_copy_atom   = Copy_Atom<UniversalCopy<uint128_t>, type>;
        using S2GCopy         = decltype(make_tiled_copy(
            s2g_copy_atom{},
            CopyThreadsLayout{},
            make_layout(Shape<_1, _8>{})
        ));
    };

    template <typename Traits, int HeadDim>
    __global__ __launch_bounds__(Traits::NumThreads)
    void forward_kernel(
        utility::Parameters<HeadDim>   params,
        typename Traits::const_pointer q,
        typename Traits::const_pointer k,
        typename Traits::const_pointer v,
        typename Traits::pointer       out
    ) {        
        using type      = typename Traits::type;
        using acc_type  = typename Traits::acc_type;
        using pointer   = typename Traits::pointer;
        using MMA       = typename Traits::MMA;
        using G2SCopy   = typename Traits::G2SCopy;
        using S2RCopyA  = typename Traits::S2RCopyA;
        using S2RCopyB  = typename Traits::S2RCopyB;
        using S2RCopyBT = typename Traits::S2RCopyBT;
        using R2SCopyC  = typename Traits::R2SCopyC;
        using S2GCopy   = typename Traits::S2GCopy;
        using CLayout   = typename Traits::mma_traits::CLayout;

        static constexpr int TileQ  = Traits::TileQ;
        static constexpr int TileKV = Traits::TileKV;
        static constexpr int Stage  = Traits::Stage;
        static constexpr int ThreadPerRow = size<0, 0>(CLayout{});
        static constexpr int RowPerThread = size<1, 1>(CLayout{}) * TileQ / Traits::PM;
        static constexpr int SwizzleBits  = 3;
        static constexpr int SwizzleBase  = utility::log2<16 / sizeof(type)>();
        static constexpr int SwizzleShift = 3;
        static constexpr int BlockKSmem   = HeadDim % 64 == 0 ? 64 : 32;
        static constexpr int SqrtD        = utility::sqrt<HeadDim>();
        
        auto SmemLayoutAtom   = make_layout(Shape<_8, Int<BlockKSmem>>{}, Stride<Int<BlockKSmem>, _1>{});
        auto SmemLayoutVTAtom = make_layout(Shape<Int<BlockKSmem>, _8>{}, Stride<_1, Int<BlockKSmem>>{});
        auto SmemSwizzleAtom = composition(
            Swizzle<SwizzleBits, SwizzleBase, SwizzleShift>{}, 
            SmemLayoutAtom
        );
        auto SmemSwizzleVTAtom = composition(
            Swizzle<SwizzleBits, SwizzleBase, SwizzleShift>{}, 
            SmemLayoutVTAtom
        );
        auto SmemLayoutQ = tile_to_shape(
            SmemSwizzleAtom,
            make_shape(Int<TileQ>{}, Int<HeadDim>{})
        );
        auto SmemLayoutK = tile_to_shape(
            SmemSwizzleAtom,
            make_shape(Int<TileKV>{}, Int<HeadDim>{}, Int<Stage>{})
        ); 
        auto SmemLayoutV = tile_to_shape(
            SmemSwizzleAtom,
            make_shape(Int<TileKV>{}, Int<HeadDim>{}, Int<Stage>{})
        );
        auto SmemLayoutO = tile_to_shape(
            SmemSwizzleAtom,
            make_shape(Int<TileQ>{}, Int<HeadDim>{})
        );
        auto SmemLayoutVT = tile_to_shape(
            SmemSwizzleVTAtom,
            make_shape(Int<HeadDim>{}, Int<TileKV>{}, Int<Stage>{})
        );
        auto SmemLayoutVTNoSwizzle = tile_to_shape(
            SmemLayoutVTAtom,
            make_shape(Int<HeadDim>{}, Int<TileKV>{}, Int<Stage>{})
        );

        extern __shared__ char shared_mem[];

        Tensor sQ = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutQ);
        // (TileQ, HeadDim)
        Tensor sK = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ)), SmemLayoutK);
        // (TileK, HeadDim, Stage)
        Tensor sV = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ) + cosize(SmemLayoutK)), SmemLayoutV);
        // (TileV, HeadDim, Stage)
        Tensor sVt = make_tensor(sV.data(), SmemLayoutVT); 
        // (HeadDim, TileKV, Stage)
        Tensor sVtNoSwizzle = make_tensor(sV.data(), SmemLayoutVTNoSwizzle); 
        // (HeadDim, TileKV, Stage)

        auto mx  = make_tensor<acc_type>(Shape<Int<RowPerThread>, _2>{}); // (RowPerThread, 2) for old_max, new_max
        fill(mx, numeric_limits<acc_type>::lowest());
        auto den = make_tensor<acc_type>(Shape<Int<RowPerThread>>{});     // (RowPerThread)  
        clear(den);

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

        MMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
        Tensor tSrQ = thr_mma.partition_fragment_A(gQ);                                  // (MMA, MMA_TileQ, MMA_HeadDim)
        Tensor tSrK = thr_mma.partition_fragment_B(gK(_, _, 0));                         // (MMA, MMA_TileK, MMA_HeadDim)
        Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<TileQ>, Int<TileKV>>{}); // (MMA, MMA_TileQ, MMA_TileK)
        // clear in for loop

        auto tOrP  = make_tensor_like<type>(tSrQ);                          // (MMA, MMA_TileQ, MMA_TileK)
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

        copy(g2s_copy_q, tSgQ_g2s_view, tSsQ_g2s_view);
        cp_async_fence();
        CUTE_UNROLL
        for (; smem_pipe_write < Stage - 1; global_read++, smem_pipe_write++) {
            copy(g2s_copy_k, tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write));
            copy(g2s_copy_v, tOgV_g2s_view(_, _, _, global_read), tOsV_g2s_view(_, _, _, smem_pipe_write));
            cp_async_fence();
        }
        cp_async_wait<Stage - 2>(); // Q is ready
        __syncthreads();

        CUTE_UNROLL
        for (int kv_idx = 0, mx_idx = 0; kv_idx < params.seq_len / TileKV; kv_idx++, mx_idx ^= 1) {
            clear(tSrS);
            // Q * K -> S
            if (kv_idx == 0) {
                copy(s2r_copy_q, tSsQ_s2r_view(_, _, 0), tSrQ_s2r_view(_, _, 0));
            }
            copy(s2r_copy_k, tSsK_s2r_view(_, _, 0, smem_pipe_read), tSrK_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrQ); i++) {
                if (i == 0) {
                    if (global_read < params.seq_len / TileKV) {
                        copy(g2s_copy_k, tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write));
                        copy(g2s_copy_v, tOgV_g2s_view(_, _, _, global_read), tOsV_g2s_view(_, _, _, smem_pipe_write));
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
            
            // SoftMax on S -> P
            CUTE_UNROLL
            for (int i = 0; i < size<0>(mx); i++) {
                int a = i % size<0, 1>(tSrS), b = i / size<0, 1>(tSrS);
                acc_type new_max = mx(i, mx_idx);
                CUTE_UNROLL
                for (int k = 0; k < size<2>(tSrS); k++)  {
                    CUTE_UNROLL
                    for (int j = 0; j < size<0, 0>(tSrS); j++) {
                        new_max = max(new_max, tSrS(make_coord(j, a), b, k));
                    }
                }
                CUTE_UNROLL
                for (int offset = ThreadPerRow >> 1; offset; offset >>= 1) {
                    new_max = max(new_max, __shfl_xor_sync(0xffffffff, new_max, offset));
                }
                mx(i, mx_idx ^ 1) = new_max;

                acc_type delta_den = 0.0;
                cute::transform(
                    tSrS(make_coord(_, a), b, _),
                    [new_max, &delta_den] __device__ (acc_type ele) { 
                        acc_type exp_ele = expf((ele - new_max) / SqrtD);
                        delta_den += exp_ele;
                        return exp_ele;
                    }
                );
                CUTE_UNROLL
                for (int offset = ThreadPerRow >> 1; offset; offset >>= 1) {
                    delta_den += __shfl_xor_sync(0xffffffff, delta_den, offset);
                }
                if (kv_idx) {
                    acc_type rescale = expf((mx(i, mx_idx) - new_max) / SqrtD);
                    cute::transform(
                        tOrO(make_coord(_, a), b, _),
                        [rescale] __device__ (acc_type ele) {
                            return ele * rescale;
                        }
                    );
                    den(i) *= rescale;
                }
                den(i) += delta_den;
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
        for (int i = 0; i < size<0>(mx); i++) {
            int a = i % size<0, 1>(tOrO), b = i / size<0, 1>(tOrO);
            cute::transform(
                tOrO(make_coord(_, a), b, _),
                [den = den(i)] __device__ (acc_type ele) {
                    return ele / den;
                }
            );
        }

        Tensor rO = make_tensor(tOrP.data(), tOrO.layout());                      // acc_type -> type
        Tensor sO = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutO); // (TileQ, HeadDim)
        copy(tOrO, rO);
        R2SCopyC r2s_copy_o;
        ThrCopy thr_r2s_copy_o = r2s_copy_o.get_slice(threadIdx.x);
        Tensor rO_r2s_view = thr_r2s_copy_o.retile_S(rO);  // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor sO_r2s_view = thr_r2s_copy_o.partition_D(sO); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads(); // make sure nobody using sO
        copy(r2s_copy_o, rO_r2s_view, sO_r2s_view);
        S2GCopy s2g_copy_o;
        ThrCopy thr_s2g_copy_o = s2g_copy_o.get_slice(threadIdx.x);
        Tensor sO_s2g_view = thr_s2g_copy_o.partition_S(sO); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor gO_s2g_view = thr_s2g_copy_o.partition_D(gO); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads();
        copy(s2g_copy_o, sO_s2g_view, gO_s2g_view);
    }

    template <size_t HeadDim>
    torch::Tensor forward_launch(
        utility::Parameters<HeadDim> params, 
        torch::Tensor q, 
        torch::Tensor k, 
        torch::Tensor v
    ) {
        using Traits = V2::Traits<>;
        static constexpr int TileQ  = Traits::TileQ;
        static constexpr int TileKV = Traits::TileKV;
        static constexpr int Stage  = Traits::Stage;
        
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
        torch::Tensor out = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        dim3 grid(params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ));
        dim3 block(cute::size(Traits::MMA{}));
        size_t shared_mem_size = (TileQ * params.head_dim + 2 * TileKV * params.head_dim * Stage) * sizeof(Traits::type);

        CUTE_CHECK_ERROR(cudaFuncSetAttribute(
            forward_kernel<Traits, HeadDim>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            shared_mem_size
        ));
        forward_kernel<Traits, HeadDim><<<grid, block, shared_mem_size>>>(
            params,
            (Traits::const_pointer)q.data_ptr<Traits::torch_type>(),
            (Traits::const_pointer)k.data_ptr<Traits::torch_type>(),
            (Traits::const_pointer)v.data_ptr<Traits::torch_type>(),
            (Traits::pointer)out.data_ptr<Traits::torch_type>()
        );
        CUTE_CHECK_ERROR(cudaDeviceSynchronize());

        return out;
    }
}