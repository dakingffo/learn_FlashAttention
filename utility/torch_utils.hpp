#pragma once
#include <torch/torch.h>
#include <tuple>

namespace FlashAttention::utility {
    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k);

    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k, torch::Tensor v);

    torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
    
    std::tuple<torch::Tensor, torch::Tensor> naive_attention_with_LSE(
        torch::Tensor q, torch::Tensor k, torch::Tensor v
    );

    bool check_equal(torch::Tensor t1, torch::Tensor t2, float tol = 0.025, int print_count = 10);
}