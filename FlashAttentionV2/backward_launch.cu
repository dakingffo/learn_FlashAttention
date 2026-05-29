#include <tuple>

#include <utility/params.hpp>
#include <utility/dispatch.hpp>

#include "traits.hpp"
#include "backward_kernel.hpp"

namespace FlashAttention::V2 {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> backward_launch(
        utility::Params params,
        torch::Tensor   q,
        torch::Tensor   k,
        torch::Tensor   v,
        torch::Tensor   out,
        torch::Tensor   lse,
        torch::Tensor   grad_o
    )   DISPATCH_VALUE(int, HeadDim, params.head_dim,
        DISPATCH_TYPE(CutlassType, q.scalar_type(), {
        using Traits = typename DispatchTraits<CutlassType, HeadDim>::type;
        static constexpr int TileQ  = Traits::TileQ;
        static constexpr int TileKV = Traits::TileKV;
        static constexpr int Stage  = Traits::Stage;

        auto out_options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
        torch::Tensor grad_q = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, out_options);
        torch::Tensor grad_k = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, out_options);
        torch::Tensor grad_v = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, out_options);
        auto d_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
        torch::Tensor d = torch::empty({params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ) * TileQ}, d_options);
        params.scale = 1 / std::sqrt(HeadDim);

        // ============================================================================
        // Pass 1: compute dQ
        //   Grid: (batch, head, ceil_div(seq_len, TileQ))
        //   Each CTA computes one TileQ x HeadDim tile of dQ:
        //     - Load dO tile, O tile into shared memory
        //          D = dO * O (vecmul for each row)    (D : regD,  dO: regdO, O : reg0)
        //     - Write D to global memory
        //     - Load Q tile into shared memory
        //     - Loop over KV tiles (pipelined via async copy):
        //          Load K, V tiles into shared memory
        //          clear(S), clear(dP)
        //          dP = dO * V^T                       (dP: reg2,  dO: regdO, V : reg1)
        //          Recompute S = Q * K^T               (S : reg3,  Q : regQ,  K : reg1)
        //          P = exp(S - LSE) (for each row)     (P : reg3,  S : reg3,  L : regL)
        //          dS = P ⊙ ((dP - D) (for each row)) (dS: reg0,  P : reg3,  dP: reg2,  D : regD)
        //          dQ += dS * K                        (dQ: regdQ, dS: reg0,  K : reg1)
        //     - Finalize: write dQ to global memory
        //   dQ is accumulated per Q tile (no atomic needed).
        // ============================================================================
        {
            dim3 grid(params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ));
            dim3 block(Traits::ThreadsPerCTA);
            size_t shared_mem_size = std::max(
                1 * cute::cosize(typename Traits::SmemLayoutQO{}), 
                2 * cute::cosize(typename Traits::SmemLayoutKV{})
            ) * sizeof(typename Traits::type);

            CUTE_CHECK_ERROR(cudaFuncSetAttribute(
                backward_dq_kernel<Traits>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_size
            ));
            backward_dq_kernel<Traits><<<grid, block, shared_mem_size>>>(
                params,
                (typename Traits::const_pointer)q.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)k.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)v.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)out.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_lse_pointer)lse.data_ptr<typename Traits::lse_type>(),
                (typename Traits::const_pointer)grad_o.data_ptr<typename Traits::torch_type>(),
                (typename Traits::pointer)grad_q.data_ptr<typename Traits::torch_type>(),
                (typename Traits::lse_pointer)d.data_ptr<typename Traits::lse_type>()
            );
            CUTE_CHECK_ERROR(cudaDeviceSynchronize());
        }

        // ============================================================================
        // Pass 2: compute dK and dV
        //   Grid: (batch, head, ceil_div(seq_len, TileKV))
        //   Each CTA computes one TileKV x HeadDim tile of dK and dV:
        //     - Load K, V tiles, D, LSE into shared memory
        //     - loop over Q, dO tiles (pipelined via async copy):
        //          recompute S = Q * K^T               (S : reg1,  Q : reg0,  K : regK)
        //          P = exp(S - LSE) (for each row)     (P : reg1,  S : reg1,  L : regL)
        //          store P into shared memory, but also keep in register
        //          dP = dO * V^T                       (dP: reg2,  dO: reg0,  V : regV)
        //          load P^T into register 
        //          dS = P ⊙ ((dP - D) (for each row)) (dS: reg1,  P : reg1,  dP: reg2,  D : regD)
        //          store dS into shared memory
        //          dV += P^T * dO                      (dV: regdV, P : reg0,  dO: reg2)
        //          load dS^T into register
        //          dK += dS^T * Q                      (dK: regdK, dS: reg1,  Q : reg2)
        //     - Finalize: write dK, dV to global memory
        //   dK and dV is accumulated per KV tile (no atomic needed).
        // ============================================================================
        {
            dim3 grid(params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileKV));
            dim3 block(Traits::ThreadsPerCTA);
            size_t shared_mem_size = (1 * cute::cosize(typename Traits::SmemLayoutSP{}) + std::max(
                1 * cute::cosize(typename Traits::SmemLayoutQO{}), 
                2 * cute::cosize(typename Traits::SmemLayoutKV{})
            ))* sizeof(typename Traits::type);

            CUTE_CHECK_ERROR(cudaFuncSetAttribute(
                backward_dkdv_kernel<Traits>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem_size
            ));
            backward_dkdv_kernel<Traits><<<grid, block, shared_mem_size>>>(
                params,
                (typename Traits::const_pointer)q.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)k.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)v.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_pointer)out.data_ptr<typename Traits::torch_type>(),
                (typename Traits::const_lse_pointer)lse.data_ptr<typename Traits::lse_type>(),
                (typename Traits::const_lse_pointer)d.data_ptr<typename Traits::lse_type>(),
                (typename Traits::const_pointer)grad_o.data_ptr<typename Traits::torch_type>(),
                (typename Traits::pointer)grad_k.data_ptr<typename Traits::torch_type>(),
                (typename Traits::pointer)grad_v.data_ptr<typename Traits::torch_type>()
            );
            CUTE_CHECK_ERROR(cudaDeviceSynchronize());
        }

        return std::make_tuple(grad_q, grad_k, grad_v);
    }))
}
