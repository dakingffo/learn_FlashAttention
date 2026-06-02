#include <utility/timer.hpp>
#include <utility/torch_utils.hpp>

#include "flash_attention_v2.hpp"

using namespace FlashAttention;
using namespace FlashAttention::utility;

static constexpr int num_tests = 4;

static const Params params[num_tests] = {
    {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=64},
    {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=64},
    {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=128},
    {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=128}
};

static const torch::TensorOptions options[num_tests] = {
    torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
    torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA),
    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)
};

void test_forward() {
    std::cout << "=== Forward Test ===" << std::endl;
    for (int i = 0; i < num_tests; i++) {
        printf("[%s] Params{.batch_size=%ld, .num_heads=%ld, .seq_len=%ld, .head_dim=%ld}:\n",
            (options[i].dtype() == torch::kFloat16 ? "FP16" : "BF16"),
            params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim
        );

        torch::Tensor q = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor k = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor v = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);

        auto [naive_out, naive_lse] = naive_attention_with_LSE(q, k, v);
        auto [flash_out, flash_lse] = V2::forward_launch(params[i], q, k, v);

        bool out_equal = utility::check_equal(naive_out, flash_out);
        bool lse_equal = utility::check_equal(naive_lse, flash_lse);

        std::cout << "Output: " << (out_equal ? "PASS" : "FAIL") << std::endl;
        std::cout << "LSE: " << (lse_equal ? "PASS" : "FAIL") << std::endl;

        if (out_equal && lse_equal) {
            TIMING("FlashAttentionV2", config<loop<5>>) {
                std::ignore = std::get<0>(V2::forward_launch(params[i], q, k, v));
            };
        }
    }
}

void test_backward() {
    std::cout << "=== Backward Test ===" << std::endl;
    for (int i = 0; i < num_tests; i++) {
        printf("[%s] Params{.batch_size=%ld, .num_heads=%ld, .seq_len=%ld, .head_dim=%ld}:\n",
            (options[i].dtype() == torch::kFloat16 ? "FP16" : "BF16"),
            params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim
        );

        torch::Tensor q = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor k = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor v = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        q.requires_grad_(true);
        k.requires_grad_(true);
        v.requires_grad_(true);

        torch::Tensor naive_out = naive_attention(q, k, v);
        naive_out.sum().backward();
        auto naive_dq = q.grad().clone();
        auto naive_dk = k.grad().clone();
        auto naive_dv = v.grad().clone();

        auto [flash_out, flash_lse] = V2::forward_launch(params[i], q.detach(), k.detach(), v.detach());
        torch::Tensor grad_o = torch::ones_like(flash_out);
        auto [flash_dq, flash_dk, flash_dv] = V2::backward_launch(
            params[i], q.detach(), k.detach(), v.detach(), flash_out, flash_lse, grad_o
        );
        bool q_equal = utility::check_equal(naive_dq, flash_dq);
        bool k_equal = utility::check_equal(naive_dk, flash_dk);
        bool v_equal = utility::check_equal(naive_dv, flash_dv);
        std::cout << "dQ: " << (q_equal ? "PASS" : "FAIL") << std::endl;
        std::cout << "dK: " << (k_equal ? "PASS" : "FAIL") << std::endl;
        std::cout << "dV: " << (v_equal ? "PASS" : "FAIL") << std::endl;
        if (q_equal && k_equal && v_equal) {
            TIMING("FlashAttentionV2 Backward", config<loop<5>>) {
                std::ignore = V2::backward_launch(
                    params[i], q.detach(), k.detach(), v.detach(), flash_out, flash_lse, grad_o
                );
            };
        }
    }
}

void test_function() {
    std::cout << "=== Autograd Function Test ===" << std::endl;
    for (int i = 0; i < num_tests; i++) {
        printf("[%s] Params{.batch_size=%ld, .num_heads=%ld, .seq_len=%ld, .head_dim=%ld}:\n",
            (options[i].dtype() == torch::kFloat16 ? "FP16" : "BF16"),
            params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim
        );

        torch::Tensor q_base = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor k_base = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);
        torch::Tensor v_base = torch::randn({params[i].batch_size, params[i].num_heads, params[i].seq_len, params[i].head_dim}, options[i]);

        // Flash: full autograd through Function::apply
        torch::Tensor q = q_base.clone().requires_grad_(true);
        torch::Tensor k = k_base.clone().requires_grad_(true);
        torch::Tensor v = v_base.clone().requires_grad_(true);
        torch::Tensor flash_out = V2::Function::apply(q, k, v);
        flash_out.sum().backward();

        // Reference: naive_attention autograd (same data, independent graph)
        torch::Tensor q2 = q_base.clone().requires_grad_(true);
        torch::Tensor k2 = k_base.clone().requires_grad_(true);
        torch::Tensor v2 = v_base.clone().requires_grad_(true);
        torch::Tensor naive_out = naive_attention(q2, k2, v2);
        naive_out.sum().backward();

        bool q_ok = utility::check_equal(q2.grad(), q.grad());
        bool k_ok = utility::check_equal(k2.grad(), k.grad());
        bool v_ok = utility::check_equal(v2.grad(), v.grad());
        std::cout << "dQ: " << (q_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "dK: " << (k_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "dV: " << (v_ok ? "PASS" : "FAIL") << std::endl;
        if (q_ok && k_ok && v_ok) {
            TIMING("FlashAttentionV2 Function", config<loop<5>>) {
                torch::Tensor q_t = q_base.clone().requires_grad_(true);
                torch::Tensor k_t = k_base.clone().requires_grad_(true);
                torch::Tensor v_t = v_base.clone().requires_grad_(true);
                V2::Function::apply(q_t, k_t, v_t).sum().backward();
            };
        }
    }
}

int main() {
    test_forward();
    test_backward();
    test_function();
    return 0;
}
