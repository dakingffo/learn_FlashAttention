#include <tuple>

#include <utility/dispatch.hpp>

#include "traits.hpp"
#include "kernel_impl.hpp"

namespace FlashAttention::V2 {
    std::tuple<torch::Tensor, torch::Tensor> forward_launch(
        utility::Params params, 
        torch::Tensor   q, 
        torch::Tensor   k, 
        torch::Tensor   v
    ) DISPATCH_VALUE(int, HeadDim, params.head_dim,
      DISPATCH_TYPE(CutlassType, q.scalar_type(), {
        using Traits = typename DispatchTraits<CutlassType, HeadDim>::type;
        static constexpr int TileQ  = Traits::TileQ;
        static constexpr int TileKV = Traits::TileKV;
        static constexpr int Stage  = Traits::Stage;
        
        auto out_options = torch::TensorOptions().dtype(q.dtype()).device(q.device());
        torch::Tensor out = torch::empty({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, out_options);
        auto lse_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
        torch::Tensor lse = torch::empty({params.batch_size, params.num_heads, params.seq_len}, lse_options);

        dim3 grid(params.batch_size, params.num_heads, cute::ceil_div(params.seq_len, TileQ));
        dim3 block(Traits::ThreadsPerCTA);
        size_t shared_mem_size = (TileQ * params.head_dim + 2 * TileKV * params.head_dim * Stage) * sizeof(typename Traits::type);

        params.scale = 1 / std::sqrt(HeadDim);
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(
            forward_kernel<Traits, HeadDim>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            shared_mem_size
        ));
        forward_kernel<Traits, HeadDim><<<grid, block, shared_mem_size>>>(
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
/*
    class Function : public torch::autograd::Function<V2::Function> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext *ctx, 
            torch::Tensor q, 
            torch::Tensor k, 
            torch::Tensor v
        ) {
            auto [out, lse] = forward_launch(params, q, k, v);

            ctx->save_for_backward({q, k, v, out, lse});
            ctx->saved_data["batch_size"] = params.batch_size;
            ctx->saved_data["num_heads"]  = params.num_heads;

            return out; 
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx, 
            torch::autograd::variable_list grad_outputs
        ) {
            torch::Tensor grad_out = grad_outputs[0];

            auto saved = ctx->get_saved_variables();
            torch::Tensor q   = saved[0];
            torch::Tensor k   = saved[1];
            torch::Tensor v   = saved[2];
            torch::Tensor out = saved[3];
            torch::Tensor lse = saved[4];

            int batch_size = ctx->saved_data["batch_size"].toInt();
            int num_heads  = ctx->saved_data["num_heads"].toInt();

            torch::Tensor dq = torch::empty_like(q);
            torch::Tensor dk = torch::empty_like(k);
            torch::Tensor dv = torch::empty_like(v);

            return {torch::Tensor(), dq, dk, dv}; 
        }
    };
*/
}