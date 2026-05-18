#pragma once
#include <torch/torch.h>

namespace FlashAttention::utility {
    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k);

    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k, torch::Tensor v);

    torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);

    bool check_equal(torch::Tensor t1, torch::Tensor t2, float tol = 0.025, int print_count = 10);
}