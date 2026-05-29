#include <cute/tensor.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
#include <cute/algorithm/tensor_reduce.hpp>
#include <cutlass/numeric_conversion.h>

#include <utility/boundary_algorithm.hpp>
#include <utility/params.hpp>

namespace FlashAttention::V2 {
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
        static constexpr int ThreadsPerCTA  = Traits::ThreadsPerCTA;

        using SmemLayoutQ           = typename Traits::SmemLayoutQO;
        using SmemLayoutK           = typename Traits::SmemLayoutKV;
        using SmemLayoutV           = typename Traits::SmemLayoutKV;
        using SmemLayoutO           = typename Traits::SmemLayoutQO;
        using SmemLayoutdO          = typename Traits::SmemLayoutQO;
        using SmemLayoutdQ          = typename Traits::SmemLayoutQO;
        using SmemLayoutKT          = typename Traits::SmemLayoutKVT;
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
        Tensor O = make_tensor(make_gmem_ptr(out), GmemLayout)(blockIdx.x, blockIdx.y, _, _);     // (seq_len, HeadDim)
        Tensor dO = make_tensor(make_gmem_ptr(grad_o), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)

        Tensor gO = local_tile(O, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0));   // (TileQ, HeadDim)
        Tensor gdO = local_tile(dO, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileQ, HeadDim)

        auto iO = make_identity_tensor(shape(gO));

        MMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
        auto tDrO = thr_mma.partition_fragment_A(gO);    // (MMA, MMA_TileQ, MMA_HeadDim)
        auto tDrdO = thr_mma.partition_fragment_A(gdO);  // (MMA, MMA_TileQ, MMA_HeadDim)

        G2SCopy g2s_copy_o;
        ThrCopy thr_g2s_copy_o = g2s_copy_o.get_slice(threadIdx.x);
        S2RCopyA s2r_copy_o;
        ThrCopy thr_s2r_copy_o = s2r_copy_o.get_slice(threadIdx.x);

        utility::copy_within_boundary<0, TileQ>(
            g2s_copy_o, params.seq_len, blockIdx.z, thr_g2s_copy_o.partition_S(iO),
            thr_g2s_copy_o.partition_S(gO), thr_g2s_copy_o.partition_D(sO)
        );
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        copy(s2r_copy_o, thr_s2r_copy_o.partition_S(sO), thr_s2r_copy_o.retile_D(tDrO));
        
        utility::copy_within_boundary<0, TileQ>(
            g2s_copy_o, params.seq_len, blockIdx.z, thr_g2s_copy_o.partition_S(iO),
            thr_g2s_copy_o.partition_S(gdO), thr_g2s_copy_o.partition_D(sdO)
        );
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

        int d_len = ceil_div(params.seq_len, TileQ) * TileQ;
        Layout GmemLayoutD = make_layout(
            make_shape(params.batch_size, params.num_heads, d_len),
            make_stride(params.num_heads * d_len, d_len, _1{})
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
        for (int i = threadIdx.x; i < TileQ; i += ThreadsPerCTA) {
            gD(i) = sD(i);
            sLSE(i) = (blockIdx.z * TileQ + i < params.seq_len ? gLSE(i) : 0.0f);
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

        // dP = dO * V^T         (dP: reg2, dO: regdO, V : reg1)
        auto& tdPrdO = tDrdO;                                               // (MMA, MMA_TileQ, MMA_HeadDim)
        auto  tdPrV  = thr_mma.partition_fragment_B(gV(_, _, 0));           // (MMA, MMA_TileV, MMA_HeadDim)
        auto  tdPrdP = thr_mma.partition_fragment_C(make_tensor<acc_type>(Shape<Int<TileQ>, Int<TileKV>>{})); 
        // tdPrdP clear in loop                                             // (MMA, MMA_TileQ, MMA_TileV)
        
        // S = Q * K^T           (S : reg3, Q:  regQ,  K : reg1)
        auto tSrQ = thr_mma.partition_fragment_A(gQ);                     // (MMA, MMA_TileQ, MMA_HeadDim)     
        auto tSrK = make_tensor(tdPrV.data(), thr_mma.partition_fragment_B(gK(_, _, 0)).layout());    
                                                                            // (MMA, MMA_TileK, MMA_HeadDim)  
        auto tSrS = thr_mma.partition_fragment_C(make_tensor<acc_type>(Shape<Int<TileQ>, Int<TileKV>>{})); 
        // tSrS clear in loop                                               // (MMA, MMA_TileQ, MMA_TileK)

        // P = exp(S - LSE)      (P : reg3, S:  reg3,  L : regL)
        // dS = P ⊙ (dP - D)    (dS: reg0, P:  reg3,  dP: reg2, D: regD)
        auto& tdSrS  = tSrS;
        auto& tdSrdP = tdPrdP;
        auto  tdSrdS = make_tensor(tDrO.data(), tdSrdP.layout());    

        // dQ += dS * K          (dQ: regdQ, dS: reg0, K : reg1)        
        auto tdQrdS = make_tensor(tdSrdS.data(), thr_mma.partition_fragment_A(make_tensor<type>(Shape<Int<TileQ>, Int<TileKV>>{})).layout());  
                                                                            // (MMA, MMA_TileQ, MMA_TileK)
        auto tdQrKt = make_tensor(tSrK.data(), thr_mma.partition_fragment_B(sKtNoSwizzle(_, _, 0)).layout());  
                                                                            // (MMA, MMA_HeadDim, MMA_TileK)  
        auto tdQrdQ = thr_mma.partition_fragment_C(gdQ);                    // (MMA, MMA_TileQ, MMA_HeadDim)
        clear(tdQrdQ);

        G2SCopy g2s_copy_q, g2s_copy_k, g2s_copy_v;

        ThrCopy thr_g2s_copy_q = g2s_copy_q.get_slice(threadIdx.x);
        utility::copy_within_boundary<0, TileQ>(
            g2s_copy_q, params.seq_len, blockIdx.z, thr_g2s_copy_q.partition_S(iQ), 
            thr_g2s_copy_q.partition_S(gQ), thr_g2s_copy_q.partition_D(sQ)
        );
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        S2RCopyA s2r_copy_q;
        ThrCopy thr_s2r_copy_q = s2r_copy_q.get_slice(threadIdx.x);
        copy(s2r_copy_q, thr_s2r_copy_q.partition_S(sQ), thr_s2r_copy_q.retile_D(tSrQ));

        ThrCopy thr_g2s_copy_k = g2s_copy_k.get_slice(threadIdx.x);
        auto tSgK_g2s_view = thr_g2s_copy_k.partition_S(gK); // (COPY, COPY_TileK, COPY_HeadDim, num_tiles_k)
        auto tSsK_g2s_view = thr_g2s_copy_k.partition_D(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)

        ThrCopy thr_g2s_copy_v = g2s_copy_v.get_slice(threadIdx.x);
        auto tdPgV_g2s_view = thr_g2s_copy_v.partition_S(gV); // (COPY, COPY_TileV, COPY_HeadDim, num_tiles_v)
        auto tdPsV_g2s_view = thr_g2s_copy_v.partition_D(sV); // (COPY, COPY_TileV, COPY_HeadDim, Stage)

        S2RCopyB s2r_copy_v;
        ThrCopy thr_s2r_copy_v = s2r_copy_v.get_slice(threadIdx.x);
        auto tdPsV_s2r_view = thr_s2r_copy_v.partition_S(sV); // (COPY, COPY_TileV, COPY_HeadDim, Stage)
        auto tdPrV_s2r_view = thr_s2r_copy_v.retile_D(tdPrV); // (COPY, COPY_TileV, COPY_HeadDim)

        S2RCopyB s2r_copy_k;
        ThrCopy thr_s2r_copy_k = s2r_copy_k.get_slice(threadIdx.x);
        auto tSsK_s2r_view = thr_s2r_copy_k.partition_S(sK); // (COPY, COPY_TileK, COPY_HeadDim, Stage)
        auto tSrK_s2r_view = thr_s2r_copy_k.retile_D(tSrK);  // (COPY, COPY_TileK, COPY_HeadDim)

        S2RCopyBT s2r_copy_kt;
        ThrCopy thr_s2r_copy_kt = s2r_copy_kt.get_slice(threadIdx.x);
        auto tdQsKt_s2r_view = thr_s2r_copy_kt.partition_S(sKt); // (COPY, COPY_HeadDim, COPY_TileV, Stage)
        auto tdQrKt_s2r_view = thr_s2r_copy_kt.retile_D(tdQrKt); // (COPY, COPY_HeadDim, COPY_TileV)

        int global_read = 0, smem_pipe_read = 0, smem_pipe_write = 0;

        CUTE_UNROLL
        for (; smem_pipe_write < Stage - 1; global_read++, smem_pipe_write++) {
            if (global_read < ceil_div(params.seq_len, TileKV)) {
                utility::copy_within_boundary<0, TileKV>(
                    g2s_copy_v, params.seq_len, global_read, thr_g2s_copy_v.partition_S(iV),
                    tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write)
                );
                utility::copy_within_boundary<0, TileKV>(
                    g2s_copy_k, params.seq_len, global_read, thr_g2s_copy_k.partition_S(iK),
                    tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
                );
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
                        utility::copy_within_boundary<0, TileKV>(
                            g2s_copy_v, params.seq_len, global_read, thr_g2s_copy_v.partition_S(iV),
                            tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write)
                        );
                        utility::copy_within_boundary<0, TileKV>(
                            g2s_copy_k, params.seq_len, global_read, thr_g2s_copy_k.partition_S(iK),
                            tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
                        );
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
                    [lse = tLSE(i), d = tD(i), scale = params.scale] (acc_type s, acc_type dp) {
                        return type(expf(s * scale - lse) * (dp - d) * scale /* for dQ += dS * K */);
                    }
                );
            }
            // dQ += dS * K
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

        auto rdQ = make_tensor(tSrQ.data(), tdQrdQ.layout());                       // acc_type -> type
        auto sdQ = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutdQ{}); // (TileQ, HeadDim)
        copy(tdQrdQ, rdQ);
        R2SCopyC r2s_copy_dq;
        ThrCopy thr_r2s_copy_dq = r2s_copy_dq.get_slice(threadIdx.x);
        __syncthreads();
        copy(r2s_copy_dq, thr_r2s_copy_dq.retile_S(rdQ), thr_r2s_copy_dq.partition_D(sdQ));
        S2GCopy s2g_copy_dq;
        ThrCopy thr_s2g_copy_dq = s2g_copy_dq.get_slice(threadIdx.x);
        __syncthreads();
        utility::copy_within_boundary<0, TileQ, false>(
            s2g_copy_dq, params.seq_len, blockIdx.z, thr_s2g_copy_dq.partition_D(iQ), 
            thr_s2g_copy_dq.partition_S(sdQ), thr_s2g_copy_dq.partition_D(gdQ)
        );
    }

    template <typename Traits>
    __global__ __launch_bounds__(Traits::ThreadsPerCTA)
    void backward_dkdv_kernel(
        __grid_constant__ const utility::Params params,
        typename Traits::const_pointer          q,
        typename Traits::const_pointer          k,
        typename Traits::const_pointer          v,
        typename Traits::const_lse_pointer      lse,
        typename Traits::const_lse_pointer      d,
        typename Traits::const_pointer          grad_o,
        typename Traits::pointer                grad_k,
        typename Traits::pointer                grad_v,
        int                                     d_len
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
        using S2RCopyAT   = typename Traits::S2RCopyAT;
        using S2RCopyBT   = typename Traits::S2RCopyBT;
        using R2SCopyA    = typename Traits::R2SCopyA;
        using R2SCopyC    = typename Traits::R2SCopyC;
        using S2GCopy     = typename Traits::S2GCopy;

        static constexpr int HeadDim       = Traits::HeadDim;
        static constexpr int TileQ         = Traits::TileQ;
        static constexpr int TileKV        = Traits::TileKV;
        static constexpr int Stage         = Traits::Stage;
        static constexpr int ThreadsPerCTA = Traits::ThreadsPerCTA;

        using SmemLayoutQ            = typename Traits::SmemLayoutQO;
        using SmemLayoutK            = typename Traits::SmemLayoutKV;
        using SmemLayoutV            = typename Traits::SmemLayoutKV;
        using SmemLayoutP            = typename Traits::SmemLayoutSP;
        using SmemLayoutdO           = typename Traits::SmemLayoutQO;
        using SmemLayoutdS           = typename Traits::SmemLayoutSP;
        using SmemLayoutdK           = typename Traits::SmemLayoutKV;
        using SmemLayoutdV           = typename Traits::SmemLayoutKV;
        using SmemLayoutQT           = typename Traits::SmemLayoutQOT;
        using SmemLayoutdOT          = typename Traits::SmemLayoutQOT;
        using SmemLayoutPT           = typename Traits::SmemLayoutSPT;
        using SmemLayoutdST          = typename Traits::SmemLayoutSPT;
        using SmemLayoutQTNoSwizzle  = typename Traits::SmemLayoutQOTLogical;
        using SmemLayoutdOTNoSwizzle = typename Traits::SmemLayoutQOTLogical;
        using SmemLayoutPTNoSwizzle  = typename Traits::SmemLayoutSPTLogical;
        using SmemLayoutdSTNoSwizzle = typename Traits::SmemLayoutSPTLogical;
        
        extern __shared__ char shared_mem[];

        Tensor sQ   = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutQ{});
        // (TileQ, HeadDim, Stage)
        Tensor sdO  = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ{})), SmemLayoutdO{});
        // (TileQ, HeadDim, Stage)
        Tensor sK   = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutK{});
        // (TileKV, HeadDim)
        Tensor sV   = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutV{});
        // (TileKV, HeadDim)
        Tensor sP   = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ{}) + cosize(SmemLayoutdO{})), SmemLayoutP{});
        // (TileQ, TileKV)
        Tensor sdS  = make_tensor(make_smem_ptr((pointer)shared_mem + cosize(SmemLayoutQ{}) + cosize(SmemLayoutdO{})), SmemLayoutdS{});
        // (TileQ, TileKV)
        Tensor sPt  = make_tensor(sP.data(), SmemLayoutPT{});
        // (TileKV, TileQ)
        Tensor sdOt = make_tensor(sdO.data(), SmemLayoutdOT{});
        // (HeadDim, TileQ, Stage)
        Tensor sdSt = make_tensor(sdS.data(), SmemLayoutdST{});
        // (TileKV, TileQ)
        Tensor sQt  = make_tensor(sQ.data(), SmemLayoutQT{});
        // (HeadDim, TileQ, Stage)
        Tensor sPtNoSwizzle  = make_tensor(sP.data(), SmemLayoutPTNoSwizzle{});
        // (TileKV, TileQ)
        Tensor sdOtNoSwizzle = make_tensor(sdO.data(), SmemLayoutdOTNoSwizzle{});
        // (HeadDim, TileQ, Stage)
        Tensor sdStNoSwizzle = make_tensor(sdS.data(), SmemLayoutdSTNoSwizzle{});
        // (TileKV, TileQ)
        Tensor sQtNoSwizzle  = make_tensor(sQ.data(), SmemLayoutQTNoSwizzle{});
        // (HeadDim, TileQ, Stage)

        auto sLSE = make_tensor(make_smem_ptr(lse_pointer((pointer)shared_mem + cosize(SmemLayoutQ{}) + cosize(SmemLayoutdO{}))), Shape<Int<TileQ>>{}); 
        // (TileQ)
        auto sD   = make_tensor(make_smem_ptr(lse_pointer((pointer)shared_mem + cosize(SmemLayoutQ{}) + cosize(SmemLayoutdO{})) + cosize(sLSE.layout())), Shape<Int<TileQ>>{}); 
        // (TileQ)

        Layout GmemLayout = make_layout(
            make_shape(params.batch_size, params.num_heads, params.seq_len, Int<HeadDim>{}),
            make_stride(params.one_batch_size(), params.one_head_size(), Int<HeadDim>{}, _1{})
        );
        Tensor Q  = make_tensor(make_gmem_ptr(q), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor dO = make_tensor(make_gmem_ptr(grad_o), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)
        Tensor K  = make_tensor(make_gmem_ptr(k), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor V  = make_tensor(make_gmem_ptr(v), GmemLayout)(blockIdx.x, blockIdx.y, _, _);      // (seq_len, HeadDim)
        Tensor dK = make_tensor(make_gmem_ptr(grad_k), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)
        Tensor dV = make_tensor(make_gmem_ptr(grad_v), GmemLayout)(blockIdx.x, blockIdx.y, _, _); // (seq_len, HeadDim)

        Layout GmemLayoutD = make_layout(
            make_shape(params.batch_size, params.num_heads, d_len),
            make_stride(params.num_heads * d_len, d_len, _1{})
        );
        Tensor D = make_tensor(make_gmem_ptr(d), GmemLayoutD)(blockIdx.x, blockIdx.y, _);       // (d_len)
        Layout GmemLayoutLSE = GmemLayoutD;
        Tensor LSE = make_tensor(make_gmem_ptr(lse), GmemLayoutLSE)(blockIdx.x, blockIdx.y, _); // (d_len)

        Tensor gQ  = local_tile(Q, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(_, 0));            // (TileQ, HeadDim, num_tiles_q)
        Tensor gdO = local_tile(dO, Tile<Int<TileQ>, Int<HeadDim>>{}, make_coord(_, 0));           // (TileQ, HeadDim, num_tiles_do)
        Tensor gK  = local_tile(K, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0));  // (TileK, HeadDim)
        Tensor gV  = local_tile(V, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0));  // (TileV, HeadDim)
        Tensor gdK = local_tile(dK, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileK, HeadDim)
        Tensor gdV = local_tile(dV, Tile<Int<TileKV>, Int<HeadDim>>{}, make_coord(blockIdx.z, 0)); // (TileV, HeadDim)

        Tensor gD = local_tile(D, Tile<Int<TileQ>>{}, make_coord(_));     // (TileQ, num_tiles_d)
        Tensor gLSE = local_tile(LSE, Tile<Int<TileQ>>{}, make_coord(_)); // (TileQ, num_tiles_d)

        auto iQ = make_identity_tensor(shape(gQ));
        auto iK = make_identity_tensor(shape(gK));
        auto iV = make_identity_tensor(shape(gV));
        auto iS = make_identity_tensor(Shape<Int<TileQ>, Int<TileKV>>{});

        MMA tiled_mma;
        ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
        
        // recompute S = Q * K^T               (S : reg1,  Q : reg0,  K : regK)
        auto tSrQ = thr_mma.partition_fragment_A(gQ(_, _, 0));            // (MMA, MMA_TileQ, MMA_HeadDim)
        auto tSrK = thr_mma.partition_fragment_B(gK);                     // (MMA, MMA_TileK, MMA_HeadDim)
        auto tSrS = thr_mma.partition_fragment_C(make_tensor<acc_type>(Shape<Int<TileQ>, Int<TileKV>>{})); 
        // tSrS clear in loop                                             // (MMA, MMA_TileQ, MMA_TileK)
        auto tSiS = thr_mma.partition_C(iS);                              // (MMA, MMA_TileQ, MMA_TileK)

        static constexpr int ThreadsPerRow = size<0, 1>(tSrS) * size<1>(tSrS); 
        auto tD = make_tensor<lse_type>(Shape<Int<ThreadsPerRow>>{});
        auto tLSE = make_tensor_like<lse_type>(tD);

        // P = exp(S - LSE) (for each row)     (P : reg0,  S : reg1,  L : regL)
        auto& tPrS = tSrS;                                                // (MMA, MMA_TileQ, MMA_TileK)
        auto  tPrP = make_tensor(tSrQ.data(), tSrS.layout());             // (MMA, MMA_TileQ, MMA_TileK)

        // dP = dO * V^T                       (dP: reg1,  dO: reg2,  V : regV)
        auto  tdPrdO = thr_mma.partition_fragment_A(gdO(_, _, 0));        // (MMA, MMA_TileQ, MMA_HeadDim)
        auto  tdPrV  = thr_mma.partition_fragment_B(gV);                  // (MMA, MMA_TileV, MMA_HeadDim)
        auto& tdPrdP = tSrS;                                              // (MMA, MMA_TileQ, MMA_TileV)
        // clear tdPrdP before use in loop

        // dS = P ⊙ ((dP - D) (for each row)) (dS: reg0,  P : reg0,  dP: reg1,  D : regD)
        auto& tdSrP  = tPrP;                                              // (MMA, MMA_TileQ, MMA_TileK)  
        auto& tdSrdP = tdPrdP;                                            // (MMA, MMA_TileQ, MMA_TileK)
        auto& tdSrdS = tdSrP;                                             // (MMA, MMA_TileQ, MMA_TileK)

        // dV += P^T * dO                      (dV: regdV, P : reg2,  dO: reg3)
        auto tdVrPt = make_tensor(tdPrdO.data(), thr_mma.partition_fragment_A(sPtNoSwizzle).layout()); 
                                                                             // (MMA, MMA_TileV, MMA_TileQ)
        auto tdVrdOt = thr_mma.partition_fragment_B(sdOtNoSwizzle(_, _, 0)); // (MMA, MMA_HeadDim, MMA_TileQ)
        auto tdVrdV  = thr_mma.partition_fragment_C(gdV);                    // (MMA, MMA_TileV, MMA_HeadDim)
        clear(tdVrdV);

        // dK += dS^T * Q                      (dK: regdK, dS: reg0,  Q : reg3)
        auto tdKrdSt = make_tensor(tdSrdS.data(), thr_mma.partition_fragment_A(sdStNoSwizzle).layout()); 
                                                                            // (MMA, MMA_TileK, MMA_TileQ)
        auto tdKrQt  = make_tensor(tdVrdOt.data(), thr_mma.partition_fragment_B(sQtNoSwizzle(_, _, 0)).layout()); 
                                                                            // (MMA, MMA_HeadDim, MMA_TileQ)
        auto tdKrdK  = thr_mma.partition_fragment_C(gdK);                   // (MMA, MMA_TileK, MMA_HeadDim)
        clear(tdKrdK);

        G2SCopy g2s_copy_q, g2s_copy_k, g2s_copy_v, g2s_copy_do;

        ThrCopy thr_g2s_copy_k = g2s_copy_k.get_slice(threadIdx.x);
        copy(g2s_copy_k, thr_g2s_copy_k.partition_S(gK), thr_g2s_copy_k.partition_D(sK));
        // utility::copy_within_boundary
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        S2RCopyB s2r_copy_k;
        ThrCopy thr_s2r_copy_k = s2r_copy_k.get_slice(threadIdx.x);
        copy(s2r_copy_k, thr_s2r_copy_k.partition_S(sK), thr_s2r_copy_k.retile_D(tSrK));

        ThrCopy thr_g2s_copy_v = g2s_copy_v.get_slice(threadIdx.x);
        copy(g2s_copy_v, thr_g2s_copy_v.partition_S(gV), thr_g2s_copy_v.partition_D(sV));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        S2RCopyB s2r_copy_v;
        ThrCopy thr_s2r_copy_v = s2r_copy_v.get_slice(threadIdx.x);
        copy(s2r_copy_v, thr_s2r_copy_v.partition_S(sV), thr_s2r_copy_v.retile_D(tdPrV));

        ThrCopy thr_g2s_copy_q = g2s_copy_q.get_slice(threadIdx.x);
        auto tSgQ_g2s_view = thr_g2s_copy_q.partition_S(gQ); // (COPY, COPY_TileQ, COPY_HeadDim, num_tiles_q)
        auto tSsQ_g2s_view = thr_g2s_copy_q.partition_D(sQ); // (COPY, COPY_TileQ, COPY_HeadDim, Stage)

        ThrCopy thr_g2s_copy_do = g2s_copy_do.get_slice(threadIdx.x);
        auto tSgdO_g2s_view = thr_g2s_copy_do.partition_S(gdO); // (COPY, COPY_TileQ, COPY_HeadDim, num_tiles_do)
        auto tSsdO_g2s_view = thr_g2s_copy_do.partition_D(sdO); // (COPY, COPY_TileQ, COPY_HeadDim, Stage)

        S2RCopyA s2r_copy_q;
        ThrCopy thr_s2r_copy_q = s2r_copy_q.get_slice(threadIdx.x);
        auto tSsQ_s2r_view = thr_s2r_copy_q.partition_S(sQ); // (COPY, COPY_TileQ, COPY_HeadDim, Stage)
        auto tSrQ_s2r_view = thr_s2r_copy_q.retile_D(tSrQ);  // (COPY, COPY_TileQ, COPY_HeadDim)

        S2RCopyA s2r_copy_do;
        ThrCopy thr_s2r_copy_do = s2r_copy_do.get_slice(threadIdx.x);
        auto tSsdO_s2r_view = thr_s2r_copy_do.partition_S(gdO);  // (COPY, COPY_TileQ, COPY_HeadDim, Stage)
        auto tSrdO_s2r_view = thr_s2r_copy_do.retile_D(tdPrdO);  // (COPY, COPY_TileQ, COPY_HeadDim)

        R2SCopyC r2s_copy_p;
        ThrCopy thr_r2s_copy_p = r2s_copy_p.get_slice(threadIdx.x);
        auto tPrP_r2s_view = thr_r2s_copy_p.retile_S(tPrP);  // (COPY, COPY_TileQ, COPY_TileK)
        auto tPsP_r2s_view = thr_r2s_copy_p.partition_D(sP); // (COPY, COPY_TileQ, COPY_TileK)

        R2SCopyC r2s_copy_ds;
        ThrCopy thr_r2s_copy_ds = r2s_copy_ds.get_slice(threadIdx.x);
        auto tdSrdS_r2s_view = thr_r2s_copy_ds.retile_S(tdSrdS); // (COPY, COPY_TileQ, COPY_TileK)
        auto tdSsdS_r2s_view = thr_r2s_copy_ds.partition_D(sdS); // (COPY, COPY_TileQ, COPY_TileK)

        S2RCopyAT s2r_copy_pt;
        ThrCopy thr_s2r_copy_pt = s2r_copy_pt.get_slice(threadIdx.x);
        auto tdVsPt_s2r_view = thr_s2r_copy_pt.partition_S(sPt); // (COPY, COPY_TileV, COPY_TileQ)
        auto tdVrPt_s2r_view = thr_s2r_copy_pt.retile_D(tdVrPt); // (COPY, COPY_TileV, COPY_TileQ)

        S2RCopyBT s2r_copy_dot;
        ThrCopy thr_s2r_copy_dot = s2r_copy_dot.get_slice(threadIdx.x);
        auto tdVsdOt_s2r_view = thr_s2r_copy_dot.partition_S(sdOt); // (COPY, COPY_TileV, COPY_TileQ, Stage)
        auto tdVrdOt_s2r_view = thr_s2r_copy_dot.retile_D(tdVrdOt); // (COPY, COPY_TileV, COPY_TileQ)

        S2RCopyAT s2r_copy_dst;
        ThrCopy thr_s2r_copy_dst = s2r_copy_dst.get_slice(threadIdx.x);
        auto tdKsdSt_s2r_view = thr_s2r_copy_dst.partition_S(sdSt); // (COPY, COPY_TileK, COPY_TileQ)
        auto tdKrdSt_s2r_view = thr_s2r_copy_dst.retile_D(tdKrdSt); // (COPY, COPY_TileK, COPY_TileQ)

        S2RCopyBT s2r_copy_qt;
        ThrCopy thr_s2r_copy_qt = s2r_copy_qt.get_slice(threadIdx.x);
        auto tdKsQt_s2r_view = thr_s2r_copy_qt.partition_S(sQt); // (COPY, COPY_HeadDim, COPY_TileQ, Stage)
        auto tdKrQt_s2r_view = thr_s2r_copy_qt.retile_D(tdKrQt); // (COPY, COPY_HeadDim, COPY_TileQ)

        int global_read = 0, smem_pipe_read = 0, smem_pipe_write = 0;

        CUTE_UNROLL
        for (; smem_pipe_write < Stage - 1; global_read++, smem_pipe_write++) {
            if (global_read < ceil_div(params.seq_len, TileKV)) {
                copy(g2s_copy_q, tSgQ_g2s_view(_, _, _, global_read), tSsQ_g2s_view(_, _, _, smem_pipe_write));
                copy(g2s_copy_do, tSgdO_g2s_view(_, _, _, global_read), tSsdO_g2s_view(_, _, _, smem_pipe_write));
                // utility::copy_within_boundary<0, TileKV>(
                //     g2s_copy_v, params.seq_len, global_read, thr_g2s_copy_v.partition_S(iV),
                //     tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write)
                // );
                // utility::copy_within_boundary<0, TileKV>(
                //     g2s_copy_k, params.seq_len, global_read, thr_g2s_copy_k.partition_S(iK),
                //     tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
                // );
            }
            cp_async_fence();
        }
        for (int i = 0; i < TileQ && i < d_len; i += ThreadsPerCTA) {
            sD(i) = gD(0, i);
            sLSE(i) = gLSE(0, i);
        }
        cp_async_wait<Stage - 2>();
        __syncthreads();

        // ============================================================================
        // Pass 2: compute dK and dV
        //   Grid: (batch, head, ceil_div(seq_len, TileKV))
        //   Each CTA computes one TileKV x HeadDim tile of dK and dV:
        //     - Load K, V tiles
        //     - loop over Q, dO tiles (pipelined via async copy):
        //          recompute S = Q * K^T               (S : reg1,  Q : reg0,  K : regK)
        //          P = exp(S - LSE) (for each row)     (P : reg0,  S : reg1,  L : regL)
        //          store P into shared memory, but also keep in register
        //          dP = dO * V^T                       (dP: reg1,  dO: reg2,  V : regV)
        //          load P^T into register
        //          dS = P ⊙ ((dP - D) (for each row)) (dS: reg0,  P : reg0,  dP: reg1,  D : regD)
        //          store dS into shared memory, then load dS^T into register
        //          dV += P^T * dO                      (dV: regdV, P : reg2,  dO: reg3)
        //          dK += dS^T * Q                      (dK: regdK, dS: reg0,  Q : reg3)
        //     - Finalize: write dK, dV to global memory
        //   dK and dV is accumulated per KV tile (no atomic needed).
        // ============================================================================

        CUTE_UNROLL
        for (int q_idx = 0; q_idx < ceil_div(params.seq_len, TileQ); q_idx++) {
            clear(tSrS);
            for (int i = 0; i < size(LSE); i++) {
                int a = i % size<0, 1>(tSrS), b = i / size<0, 1>(tSrS);
                int row = get<0>(tSiS(make_coord(0, a), b, 0));
                tLSE(i) = sLSE(row);
                tD(i) = sD(row);
            }
            // S = Q * K^T
            copy(s2r_copy_q, tSsQ_s2r_view(_, _, 0, smem_pipe_read), tSrQ_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tSrQ); i++) {
                if (i == 0) {
                    if (global_read < ceil_div(params.seq_len, TileKV)) {
                        copy(g2s_copy_q, tSgQ_g2s_view(_, _, _, global_read), tSsQ_g2s_view(_, _, _, smem_pipe_write));
                        copy(g2s_copy_do, tSgdO_g2s_view(_, _, _, global_read), tSsdO_g2s_view(_, _, _, smem_pipe_write));
                        // utility::copy_within_boundary<0, TileKV>(
                        //     g2s_copy_v, params.seq_len, global_read, thr_g2s_copy_v.partition_S(iV),
                        //     tdPgV_g2s_view(_, _, _, global_read), tdPsV_g2s_view(_, _, _, smem_pipe_write)
                        // );
                        // utility::copy_within_boundary<0, TileKV>(
                        //     g2s_copy_k, params.seq_len, global_read, thr_g2s_copy_k.partition_S(iK),
                        //     tSgK_g2s_view(_, _, _, global_read), tSsK_g2s_view(_, _, _, smem_pipe_write)
                        // );
                        global_read++;
                        smem_pipe_write = (smem_pipe_write + 1) % Stage;
                    }
                    cp_async_fence();
                }
                if (i + 1 < size<2>(tSrQ)) {
                    copy(s2r_copy_q, tSsQ_s2r_view(_, _, i + 1, smem_pipe_read), tSrQ_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tSrS, tSrQ(_, _, i), tSrK(_, _, 0), tSrS);
            }
            // P = exp(S - LSE) (for each row)
            for (int i = 0; i < size(LSE); i++) {
                int a = i % size<0, 1>(tSrS), b = i / size<0, 1>(tSrS);
                cute::transform(
                    tPrS(make_coord(_, a), b, _),
                    tPrP(make_coord(_, a), b, _),
                    [lse = tLSE(i), scale = params.scale] (acc_type s) {
                        return type(expf(s * scale - lse)) ;
                    }
                );
            }
            copy(r2s_copy_p, tPrP_r2s_view, tPsP_r2s_view);
            // dP = dO * V^T
            copy(s2r_copy_do, tSsdO_s2r_view(_, _, 0, smem_pipe_read), tSrdO_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tdPrdO); i++) {
                if (i + 1 < size<2>(tdPrdO)) {
                    copy(s2r_copy_do, tSsdO_s2r_view(_, _, i + 1, smem_pipe_read), tSrdO_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tdPrdP, tdPrdO(_, _, i), tdPrV(_, _, 0), tdPrdP);
            }
            // load P^T into register
            copy(s2r_copy_pt, tdVsPt_s2r_view, tdVrPt_s2r_view);
            // dS = P ⊙ ((dP - D) (for each row))
            for (int i = 0; i < size(LSE); i++) {
                int a = i % size<0, 1>(tSrS), b = i / size<0, 1>(tSrS);
                cute::transform(
                    tdSrP(make_coord(_, a), b, _),
                    tdSrdP(make_coord(_, a), b, _),
                    tdSrdS(make_coord(_, a), b, _),
                    [d = tD(i), scale = params.scale] (acc_type p, acc_type dp) {
                        return type(p * (dp - d) * scale /* for dK += dS^T * Q */);
                    }
                );
            }
            // store dS into shared memory, then load dS^T into register
            copy(r2s_copy_ds, tdSrdS_r2s_view, tdSsdS_r2s_view);
            copy(s2r_copy_dst, tdKsdSt_s2r_view, tdKrdSt_s2r_view);
            // dV += P^T * dO
            copy(s2r_copy_dot, tdVsdOt_s2r_view(_, _, 0, smem_pipe_read), tdVrdOt_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tdVrPt); i++) {
                if (i + 1 < size<2>(tdVrPt); i++) {
                    copy(s2r_copy_dot, tdVsdOt_s2r_view(_, _, i + 1, smem_pipe_read), tdVrdOt_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tdVrdV, tdVrPt(_, _, i), tdVrdOt(_, _, i), tdVrdV);
            }
            // dK += dS^T * Q
            copy(s2r_copy_qt, tdKsQt_s2r_view(_, _, 0, smem_pipe_read), tdKrQt_s2r_view(_, _, 0));
            CUTE_UNROLL
            for (int i = 0; i < size<2>(tdKrdSt); i++) {
                if (i + 1 < size<2>(tdKrdSt); i++) {
                    copy(s2r_copy_qt, tdKsQt_s2r_view(_, _, i + 1, smem_pipe_read), tdKrQt_s2r_view(_, _, i + 1));
                }
                gemm(tiled_mma, tdKrdK, tdKrdSt(_, _, i), tdKrQt(_, _, i), tdKrdK);
            }

            smem_pipe_read = (smem_pipe_read + 1) % Stage;
            if (q_idx + 1 < ceil_div(params.seq_len, TileQ)) {
                for (int i = 0; i < TileQ && q_idx * TileQ + i < d_len; i += ThreadsPerCTA) {
                    sD(i) = gD(q_idx + 1, i);
                    sLSE(i) = gLSE(q_idx + 1, i);
                }
                cp_async_wait<Stage - 2>();
                __syncthreads();
            }
        }

        auto rdK = make_tensor(tdKrdSt.data(), tdKrdK.layout()); // acc_type -> type
        auto sdK = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutdK{}); // (TileKV, HeadDim)
        copy(tdKrdK, rdK);
        R2SCopyC r2s_copy_dk;
        ThrCopy thr_r2s_copy_dk = r2s_copy_dk.get_slice(threadIdx.x);
        __syncthreads();
        copy(r2s_copy_dk, thr_r2s_copy_dk.retile_S(rdK), thr_r2s_copy_dk.partition_D(sdK));
        S2GCopy s2g_copy_dk;
        ThrCopy thr_s2g_copy_dk = s2g_copy_dk.get_slice(threadIdx.x);
        __syncthreads();
        copy(s2g_copy_dk, thr_s2g_copy_dk.partition_S(sdK), thr_s2g_copy_dk.partition_D(gdK));
        // utility::copy_within_boundary<0, TileKV, false>(

        auto rdV = make_tensor(tdVrPt.data(), tdVrdV.layout()); // acc_type -> type
        auto sdV = make_tensor(make_smem_ptr((pointer)shared_mem), SmemLayoutdV{}); // (TileKV, HeadDim)
        copy(tdVrdV, rdV);
        R2SCopyC r2s_copy_dv;
        ThrCopy thr_r2s_copy_dv = r2s_copy_dv.get_slice(threadIdx.x);
        __syncthreads();
        copy(r2s_copy_dv, thr_r2s_copy_dv.retile_S(rdV), thr_r2s_copy_dv.partition_D(sdV));
        S2GCopy s2g_copy_dv;
        ThrCopy thr_s2g_copy_dv = s2g_copy_dv.get_slice(threadIdx.x);
        __syncthreads();
        copy(s2g_copy_dv, thr_s2g_copy_dv.partition_S(sdV), thr_s2g_copy_dv.partition_D(gdV));
        // utility::copy_within_boundary<0, TileKV, false>(
    }
}