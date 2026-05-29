#include <tuple>

#include <utility/params.hpp>
#include <utility/dispatch.hpp>

#include "traits.hpp"
#include "forward_kernel.hpp"

namespace FlashAttention::V2 {
    std::tuple<torch::Tensor, torch::Tensor> forward_launch(
        utility::Params params, 
        torch::Tensor   q, 
        torch::Tensor   k, 
        torch::Tensor   v
    )   DISPATCH_VALUE(int, HeadDim, params.head_dim,
        DISPATCH_TYPE(CutlassType, q.scalar_type(), {
        using Traits = typename DispatchTraits<CutlassType, HeadDim>::QParallel;
        using SmemLayoutQO = typename Traits::SmemLayoutQO;
        using SmemLayoutKV = typename Traits::SmemLayoutKV;
        static constexpr int TileQ  = Traits::TileQ;

        auto out_options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
        torch::Tensor out = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, out_options);
        auto lse_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
        torch::Tensor lse = torch::empty({params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ) * TileQ}, lse_options);
        params.scale = 1 / std::sqrt(HeadDim);

        // ============================================================================
        // Forward pass
        //   Grid: (batch, head, ceil_div(seq_len, TileQ))
        //   Each CTA computes one TileQ x HeadDim tile of O:
        //     - Load Q tile into shared memory
        //     - Loop over KV tiles (pipelined via async copy):
        //         Load K, V tiles into shared memory
        //         clear(S)
        //         S = Q * K^T
        //         Online softmax: update running mx, den, rescale old accumulator:
        //         P = softmax(S * scale), O *= exp(mx_old - mx_new), den *= exp(mx_old - mx_new)
        //         O += P * V
        //     - Finalize: O /= den, write O and LSE to global memory
        //   O and LSE are accumulated per Q tile (no atomic needed).
        // ============================================================================
        dim3 grid(params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ));
        dim3 block(Traits::ThreadsPerCTA);
        size_t shared_mem_size = std::max(
            1 * cute::cosize(SmemLayoutQO{}), 
            2 * cute::cosize(SmemLayoutKV{})
        ) * sizeof(typename Traits::type);
        
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(
            forward_kernel<Traits>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            shared_mem_size
        ));
        forward_kernel<Traits><<<grid, block, shared_mem_size>>>(
            params,
            (typename Traits::const_pointer)q.data_ptr<typename Traits::torch_type>(),
            (typename Traits::const_pointer)k.data_ptr<typename Traits::torch_type>(),
            (typename Traits::const_pointer)v.data_ptr<typename Traits::torch_type>(),
            (typename Traits::pointer)out.data_ptr<typename Traits::torch_type>(),
            (typename Traits::lse_pointer)lse.data_ptr<typename Traits::lse_type>()
        );
        CUTE_CHECK_ERROR(cudaDeviceSynchronize());

        return std::make_tuple(out, lse);
    }))
}