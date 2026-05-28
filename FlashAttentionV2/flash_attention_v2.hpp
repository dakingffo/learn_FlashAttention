#pragma once

#include <tuple>

#include <torch/torch.h>

#include <utility/params.hpp>

namespace FlashAttention::V2 {
    std::tuple<torch::Tensor, torch::Tensor> forward_launch(
        utility::Params params, 
        torch::Tensor   q, 
        torch::Tensor   k, 
        torch::Tensor   v
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> backward_launch(
        utility::Params params,
        torch::Tensor   q,
        torch::Tensor   k,
        torch::Tensor   v,
        torch::Tensor   out,
        torch::Tensor   lse,
        torch::Tensor   grad_o
    );

    class Function : public torch::autograd::Function<V2::Function> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx, 
            torch::Tensor                     q, 
            torch::Tensor                     k, 
            torch::Tensor                     v
        ) {
            TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0));
            TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1));
            TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2));
            TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3));
            TORCH_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype());

            utility::Params params{
                .batch_size = q.size(0), 
                .num_heads  = q.size(1), 
                .seq_len    = q.size(2), 
                .head_dim   = q.size(3)
            };
            auto [out, lse] = forward_launch(params, q, k, v);

            ctx->save_for_backward({q, k, v, out, lse});

            return out; 
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx, 
            torch::autograd::variable_list grad_outputs
        ) {
            torch::Tensor grad_o = grad_outputs[0];

            auto saved = ctx->get_saved_variables();
            torch::Tensor q   = saved[0];
            torch::Tensor k   = saved[1];
            torch::Tensor v   = saved[2];
            torch::Tensor out = saved[3];
            torch::Tensor lse = saved[4];

            TORCH_CHECK(q.size(0) == grad_o.size(0));
            TORCH_CHECK(q.size(1) == grad_o.size(1));
            TORCH_CHECK(q.size(2) == grad_o.size(2));
            TORCH_CHECK(q.size(3) == grad_o.size(3));
            TORCH_CHECK(q.dtype() == grad_o.dtype());

            utility::Params params{
                .batch_size = grad_o.size(0), 
                .num_heads  = grad_o.size(1), 
                .seq_len    = grad_o.size(2), 
                .head_dim   = grad_o.size(3)
            };

            auto [grad_q, grad_k, grad_v, d] = backward_launch(params, q, k, v, out, lse, grad_o);

            return {torch::Tensor(), grad_q, grad_k, grad_v}; 
        }
    };
}