#include <cute/tensor.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cutlass/numeric_conversion.h>

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

    template <typename Traits>
    __global__ __launch_bounds__(Traits::ThreadsPerCTA)
    void backward_dq_kernel(
        __grid_constant__ const utility::Params params,
        typename Traits::const_pointer          q,
        typename Traits::const_pointer          k,
        typename Traits::const_pointer          v,
        typename Traits::const_pointer          out,
        typename Traits::const_lse_pointer      lse,
        typename Traits::const_pointer          grad_o,
        typename Traits::pointer                grad_q,
        typename Traits::lse_pointer            d
    ) {        
        using type        = typename Traits::type;
        using acc_type    = typename Traits::acc_type;
        using lse_type    = typename Traits::lse_type;
        using pointer     = typename Traits::pointer;
        using lse_pointer = typename Traits::lse_pointer;
        using MMA         = typename Traits::MMA;
        using G2SCopy     = typename Traits::G2SCopy;
        using S2RCopyA    = typename Traits::S2RCopyA;
        using S2RCopyB    = typename Traits::S2RCopyB;
        using S2RCopyBT   = typename Traits::S2RCopyBT;
        using R2SCopyC    = typename Traits::R2SCopyC;
        using S2GCopy     = typename Traits::S2GCopy;

        static constexpr int HeadDim        = Traits::HeadDim;
        static constexpr int TileQ          = Traits::TileQ;
        static constexpr int TileKV         = Traits::TileKV;
        static constexpr int Stage          = Traits::Stage;
        static constexpr int AtomM          = Traits::AtomM;
        static constexpr int PM             = Traits::PM;
        static constexpr int ThreadsPerRow  = Traits::ThreadsPerRow;
        static constexpr int RowsPerThread  = Traits::RowsPerThread;
        static constexpr int ThreadsPerCol  = Traits::ThreadsPerCol;
        static constexpr int ThreadsPerAtom = Traits::ThreadsPerAtom;

        using SmemLayoutQ           = typename Traits::SmemLayoutQO;
        using SmemLayoutK           = typename Traits::SmemLayoutKV;
        using SmemLayoutV           = typename Traits::SmemLayoutKV;
        using SmemLayoutO           = typename Traits::SmemLayoutQO;
        using SmemLayoutdO          = typename Traits::SmemLayoutQO;
        using SmemLayoutdQ          = typename Traits::SmemLayoutQO;
        using SmemLayoutKT          = typename Traits::SmemLayoutKVT;
        using SmemLayoutKNoSwizzle  = typename Traits::SmemLayoutKVLogical;
        using SmemLayoutKTNoSwizzle = typename Traits::SmemLayoutKVTLogical;
        
        extern __shared__ char shared_mem[];

        /* ------------------------------ compute D ------------------------------ */
        Tensor sO = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutO{});   
        // (TileQ, HeadDim)
        Tensor sdO = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutdO{}); 
        // (TileQ, HeadDim)

        auto tLSE = make_tensor<lse_type>(Shape<Int<RowsPerThread>>{});
        auto tD   = make_tensor<lse_type>(Shape<Int<RowsPerThread>>{});
        clear(tD);

        Layout GmemLayout = make_layout(
            make_shape(params.batch_size, params.num_heads, params.seq_len, Int<HeadDim>{}),
            make_stride(params.one_batch_size(), params.one_head_size(), Int<HeadDim>{}, _1{})
        );
        Tensor O = make_tensor(make_gmem_ptr(out), GmemLayout)(blockIdx.x, blockIdx.y, _, _);       // (seq_len, HeadDim)
        Tensor dO = make_tensor(make_gmem_ptr(grad_o), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)

        Tensor gO = local_tile(O, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0));   // (TileQ, HeadDim)
        Tensor gdO = local_tile(dO, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileQ, HeadDim)

        auto iO = make_identity_tensor(shape(gO));

        MMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
        Tensor tDrO = thr_mma.partition_fragment_A(gO);   // (MMA, MMA_TileQ, MMA_HeadDim)
        Tensor tDrdO = thr_mma.partition_fragment_A(gdO);  // (MMA, MMA_TileQ, MMA_HeadDim)

        G2SCopy g2s_copy_o;
        ThrCopy thr_g2s_copy_o = g2s_copy_o.get_slice(threadIdx.x);
        S2RCopyA s2r_copy_o;
        ThrCopy thr_s2r_copy_o = s2r_copy_o.get_slice(threadIdx.x);

        copy(g2s_copy_o, thr_g2s_copy_o.partition_S(gO), thr_g2s_copy_o.partition_D(sO));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        copy(s2r_copy_o, thr_s2r_copy_o.partition_S(sO), thr_s2r_copy_o.retile_D(tDrO));
        
        copy(g2s_copy_o, thr_g2s_copy_o.partition_S(gdO), thr_g2s_copy_o.partition_D(sdO));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        copy(s2r_copy_o, thr_s2r_copy_o.partition_S(sdO), thr_s2r_copy_o.retile_D(tDrdO));

        CUTE_UNROLL
        for (int i = 0; i < RowsPerThread; i++) {
            int a = i % size<0, 1>(tDrO), b = i / size<0, 1>(tDrO);
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tDrO); k++)  {
                CUTE_UNROLL
                for (int j = 0; j < size<0, 0>(tDrO); j++) {
                    CUTE_UNROLL
                    for (int l = 0; l < size<0, 2>(tDrO); l++) {
                        tD(i) += tDrdO(make_coord(j, a, l), b, k) * tDrO(make_coord(j, a, l), b, k);
                    }
                }
            }
            CUTE_UNROLL
            for (int offset = ThreadsPerRow >> 1; offset; offset >>= 1) {
                tD(i) += __shfl_xor_sync(0xffffffff, tD(i), offset);
            }
        }

        Layout GmemLayoutD = make_layout(
            make_shape(params.batch_size, params.num_heads, params.seq_len),
            make_stride(params.num_heads * params.seq_len, params.seq_len, _1{})
        );
        Tensor D  = make_tensor(make_gmem_ptr(d), GmemLayoutD)(blockIdx.x, blockIdx.y, _);
        Tensor gD = local_tile(D, Tile<Int<TileQ>>{}, make_coord(blockIdx.z));                // (TileQ)
        Tensor sD = make_tensor(make_smem_ptr((lse_pointer)shared_mem), Shape<Int<TileQ>>{}); // (TileQ)
        Layout GmemLayoutLSE = GmemLayoutD;
        Tensor LSE  = make_tensor(make_gmem_ptr(lse), GmemLayoutLSE)(blockIdx.x, blockIdx.y, _);
        Tensor gLSE = local_tile(LSE, Tile<Int<TileQ>>{}, make_coord(blockIdx.z));              // (TileQ)
        Tensor sLSE = make_tensor(make_smem_ptr((lse_pointer)shared_mem + cosize(sD.layout())), Shape<Int<TileQ>>{}); // (TileQ)
        __syncthreads();
        if (threadIdx.x % ThreadsPerRow == 0) {
            CUTE_UNROLL
            for (int i = 0; i < RowsPerThread; i++) {
                int a = i % size<0, 1>(tDrO), b = i / size<0, 1>(tDrO);
                int idx = b * PM + threadIdx.x / ThreadsPerAtom * AtomM + 
                          a * ThreadsPerCol + threadIdx.x % ThreadsPerAtom / ThreadsPerRow; 
                sD(idx) = tD(i);
            }
        }
        __syncthreads();
        CUTE_UNROLL
        for (int i = threadIdx.x; i < TileQ; i += Traits::ThreadsPerCTA) {
            gD(i) = sD(i);
            sLSE(i) = gLSE(i);
        }
        __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < RowsPerThread; i++) {
            int a = i % size<0, 1>(tDrO), b = i / size<0, 1>(tDrO);
            int idx = b * PM + threadIdx.x / ThreadsPerAtom * AtomM + 
                      a * ThreadsPerCol + threadIdx.x % ThreadsPerAtom / ThreadsPerRow; 
            tLSE(i) = sLSE(idx);
        }
        __syncthreads();

        /* ------------------------------ compute dQ ------------------------------ */
        Tensor sQ = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutQ{});
        // (TileQ, HeadDim)
        Tensor sK = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutK{});
        // (TileK, HeadDim, Stage)
        Tensor sV = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutK{})), SmemLayoutV{});
        // (TileV, HeadDim, Stage)
        Tensor sKt = make_tensor(sK.data(), SmemLayoutKT{}); 
        // (HeadDim, TileKV, Stage)
        Tensor sKtNoSwizzle = make_tensor(sK.data(), SmemLayoutKTNoSwizzle{}); 
        // (HeadDim, TileKV, Stage)

        Tensor Q  = make_tensor(make_gmem_ptr(q), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor K  = make_tensor(make_gmem_ptr(k), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor V  = make_tensor(make_gmem_ptr(v), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor dQ = make_tensor(make_gmem_ptr(grad_q), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)

        Tensor gQ  = local_tile(Q, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0));  // (TileQ, HeadDim)
        Tensor gK  = local_tile(K, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(_, 0));          // (TileK, HeadDim, num_tiles_k)
        Tensor gV  = local_tile(V, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(_, 0));          // (TileV, HeadDim, num_tiles_v)
        Tensor gdQ = local_tile(dQ, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileQ, HeadDim)

        auto iQ = make_identity_tensor(shape(gQ));
        auto iK = make_identity_tensor(shape(gK(_, _, 0)));
        auto iV = make_identity_tensor(shape(gV(_, _, 0)));
        auto iS = make_identity_tensor(Shape<Int<TileQ>, Int<TileKV>>{});

        // dP = dO * V^T         (dP: reg1, dO: regdO, V : reg0)
        auto& tdPrdO = tDrdO;                                              // (MMA, MMA_TileQ, MMA_HeadDim)
        auto  tdPrV  = make_tensor(tDrO.data(), thr_mma.partition_fragment_B(gV(_, _, 0)).layout());          
                                                                            // (MMA, MMA_TileV, MMA_HeadDim)
        auto  tdPrdP = thr_mma.partition_fragment_C(make_tensor<acc_type>(Shape<Int<TileQ>, Int<TileKV>>{})); 
        // tdPrdP clear in loop                                             // (MMA, MMA_TileQ, MMA_TileV)
        
        // S = Q * K^T           (S : reg2, Q:  regQ,  K : reg0)
        Tensor tSrQ = thr_mma.partition_fragment_A(gQ);                   // (MMA, MMA_TileQ, MMA_HeadDim) 
        auto   tSrK = make_tensor(tDrO.data(), thr_mma.partition_fragment_B(gK(_, _, 0)).layout());    
                                                                            // (MMA, MMA_TileK, MMA_HeadDim)  
        auto   tSrS = thr_mma.partition_fragment_C(make_tensor<acc_type>(Shape<Int<TileQ>, Int<TileKV>>{})); 
        // tSrS clear in loop                                               // (MMA, MMA_TileQ, MMA_TileK)

        // P = exp(S - LSE)      (P : reg2, S:  reg2,  L : regL)
        // dS = P ⊙ (dP - D)    (dS: reg2, P:  reg2,  dP: reg1, D: regD)
        auto& tdSrS  = tSrS;
        auto& tdSrdP = tdPrdP;
        auto& tdSrdS = tSrS;

        // dQ += dS * K          (dQ: regdQ, dS: reg3, K : reg0)
        auto tdQrdS = thr_mma.partition_fragment_A(make_tensor<type>(Shape<Int<TileQ>, Int<TileKV>>{}));  
                                                                            // (MMA, MMA_TileQ, MMA_TileK)
        auto tdQrKt = make_tensor(tSrK.data(), thr_mma.partition_fragment_B(sKtNoSwizzle(_, _, 0)).layout());  
                                                                            // (MMA, MMA_HeadDim, MMA_TileK)  
        auto tdQrdQ = thr_mma.partition_fragment_C(gdQ);                    // (MMA, MMA_TileQ, MMA_HeadDim)
        clear(tdQrdQ);

        G2SCopy g2s_copy_q, g2s_copy_k, g2s_copy_v;
        ThrCopy thr_g2s_copy_q = g2s_copy_q.get_slice(threadIdx.x);
        Tensor tSgQ_g2s_view = thr_g2s_copy_q.partition_S(gQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor tSsQ_g2s_view = thr_g2s_copy_q.partition_D(sQ); // (COPY, COPY_TileQ, COPY_HeadDim)

        ThrCopy thr_g2s_copy_k = g2s_copy_k.get_slice(threadIdx.x);
        Tensor tSgK_g2s_view = thr_g2s_copy_k.partition_S(gK); // (COPY, COPY_TileK, COPY_HeadDim, num_tiles_k)
        Tensor tSsK_g2s_view = thr_g2s_copy_k.partition_D(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)

        ThrCopy thr_g2s_copy_v = g2s_copy_v.get_slice(threadIdx.x);
        Tensor tdPgV_g2s_view = thr_g2s_copy_v.partition_S(gV); // (COPY, COPY_TileV, COPY_HeadDim, num_tiles_v)
        Tensor tdPsV_g2s_view = thr_g2s_copy_v.partition_D(sV); // (COPY, COPY_TileV, COPY_HeadDim, Stage)

        S2RCopyA s2r_copy_q;
        ThrCopy thr_s2r_copy_q = s2r_copy_q.get_slice(threadIdx.x);
        Tensor tSsQ_s2r_view = thr_s2r_copy_q.partition_S(sQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor tSrQ_s2r_view = thr_s2r_copy_q.retile_D(tSrQ);  // (COPY, COPY_TileQ, COPY_HeadDim)

        S2RCopyB s2r_copy_v;
        ThrCopy thr_s2r_copy_v = s2r_copy_v.get_slice(threadIdx.x);
        Tensor tdPsV_s2r_view = thr_s2r_copy_v.partition_S(sV);  // (COPY, COPY_TileV, COPY_HeadDim, Stage)
        auto tdPrV_s2r_view = thr_s2r_copy_v.retile_D(tdPrV);  // (COPY, COPY_TileV, COPY_HeadDim)

        S2RCopyB s2r_copy_k;
        ThrCopy thr_s2r_copy_k = s2r_copy_k.get_slice(threadIdx.x);
        Tensor tSsK_s2r_view = thr_s2r_copy_k.partition_S(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)
        auto tSrK_s2r_view = thr_s2r_copy_k.retile_D(tSrK);  // (COPY, COPY_TileK, COPY_HeadDim)

        S2RCopyBT s2r_copy_kt;
        ThrCopy thr_s2r_copy_kt = s2r_copy_kt.get_slice(threadIdx.x);
        Tensor tdQsKt_s2r_view = thr_s2r_copy_kt.partition_S(sKt);  // (COPY, COPY_HeadDim, COPY_TileV, Stage)
        auto tdQrKt_s2r_view = thr_s2r_copy_kt.retile_D(tdQrKt);  // (COPY, COPY_HeadDim, COPY_TileV)

        int global_read = 0, smem_pipe_read = 0, smem_pipe_write = 0;

        copy(g2s_copy_q, tSgQ_g2s_view, tSsQ_g2s_view);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        copy(s2r_copy_q, tSsQ_s2r_view, tSrQ_s2r_view);

        CUTE_UNROLL
        for (; smem_pipe_write < Stage - 1; global_read++, smem_pipe_write++) {
            if (global_read < ceil_div(params.seq_len, TileKV)) {
                copy(g2s_copy_v, tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write));
                copy(g2s_copy_k, tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write));
            }
            cp_async_fence();
        }
        cp_async_wait<Stage - 2>();
        __syncthreads();

        CUTE_UNROLL
        for (int kv_idx = 0; kv_idx < ceil_div(params.seq_len, TileKV); kv_idx++) {
            clear(tdPrdP);
            clear(tSrS);
            // dP = dO * V^T
            copy(s2r_copy_v, tdPsV_s2r_view(_, _, 0, smem_pipe_read), tdPrV_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tdPrdO); i++) {
                if (i == 0) {
                    if (global_read < ceil_div(params.seq_len, TileKV)) {
                        copy(g2s_copy_v, tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write));
                        copy(g2s_copy_k, tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write));
                        global_read++;
                        smem_pipe_write = (smem_pipe_write + 1) % Stage;
                    }
                    cp_async_fence();
                }
                if (i + 1 < size<2>(tdPrdO)) {
                    copy(s2r_copy_v, tdPsV_s2r_view(_, _, i + 1, smem_pipe_read), tdPrV_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tdPrdP, tdPrdO(_, _, i), tdPrV(_, _, i), tdPrdP);
            }
            // S = Q * K^T
            copy(s2r_copy_k, tSsK_s2r_view(_, _, 0, smem_pipe_read), tSrK_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrQ); i++) {
                if (i + 1 < size<2>(tSrQ)) {
                    copy(s2r_copy_k, tSsK_s2r_view(_, _, i + 1, smem_pipe_read), tSrK_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tSrS, tSrQ(_, _, i), tSrK(_, _, i), tSrS);
            }
            // P = exp(S - LSE) (for each row)   
            // dS = P ⊙ ((dP - D) (for each row)) 
            CUTE_UNROLL
            for (int i = 0; i < RowsPerThread; i++) {
                int a = i % size<0, 1>(tdSrS), b = i / size<0, 1>(tdSrS);
                cute::transform(
                    tdSrS(make_coord(_, a), b, _),
                    tdSrdP(make_coord(_, a), b, _),
                    tdSrdS(make_coord(_, a), b, _),
                    [lse = tLSE(i), d = tD(i), scale = params.scale](acc_type s, acc_type dp) {
                        return expf(s * scale - lse) * (dp - d) * scale /* for dQ += dS * K */;
                    }
                );
            }
            // dQ += dS * K
            copy(tdSrdS, tdQrdS);
            copy(s2r_copy_kt, tdQsKt_s2r_view(_, _, 0, smem_pipe_read), tdQrKt_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tdQrdS); i++) {
                if (i + 1 < size<2>(tdQrdS)) {
                    copy(s2r_copy_kt, tdQsKt_s2r_view(_, _, i + 1, smem_pipe_read), tdQrKt_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tdQrdQ, tdQrdS(_, _, i), tdQrKt(_, _, i), tdQrdQ);
            }

            smem_pipe_read = (smem_pipe_read + 1) % Stage;
            if (kv_idx + 1 < ceil_div(params.seq_len, TileKV)) {
                cp_async_wait<Stage - 2>();
                __syncthreads();
            }
        }

        auto rdQ = make_tensor(tSrQ.data(), tdQrdQ.layout());                         // acc_type -> type
        Tensor sdQ = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutdQ{}); // (TileQ, HeadDim)
        copy(tdQrdQ, rdQ);
        R2SCopyC r2s_copy_dq;
        ThrCopy thr_r2s_copy_dq = r2s_copy_dq.get_slice(threadIdx.x);
        auto rdQ_r2s_view = thr_r2s_copy_dq.retile_S(rdQ);      // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor sdQ_r2s_view = thr_r2s_copy_dq.partition_D(sdQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads();
        copy(r2s_copy_dq, rdQ_r2s_view, sdQ_r2s_view);
        S2GCopy s2g_copy_dq;
        ThrCopy thr_s2g_copy_dq = s2g_copy_dq.get_slice(threadIdx.x);
        Tensor sdQ_s2g_view = thr_s2g_copy_dq.partition_S(sdQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        Tensor gdQ_s2g_view = thr_s2g_copy_dq.partition_D(gdQ); // (COPY, COPY_TileQ, COPY_HeadDim)
        __syncthreads();
        copy(s2g_copy_dq, sdQ_s2g_view, gdQ_s2g_view);
    }
}