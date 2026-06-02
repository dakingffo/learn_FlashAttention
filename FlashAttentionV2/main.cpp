#include <utility/timer.hpp>
#include <utility/torch_utils.hpp>

#include "flash_attention_v2.hpp"

using namespace FlashAttention;
using namespace FlashAttention::utility;

void test_forward() {
    Params params[4] = {
        {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=64},
        {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=64},
        {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=128},
        {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=128}
    };
    torch::TensorOptions options[4] = {
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)
    };

    std::cout << "=== Forward Test ===" << std::endl;
    for (int i = 0; i < 4; i++) {
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
    Params params[4] = {
        {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=64},
        {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=64},
        {.batch_size=4, .num_heads=64, .seq_len=2048, .head_dim=128},
        {.batch_size=4, .num_heads=64, .seq_len=2000, .head_dim=128}
    };
    torch::TensorOptions options[4] = {
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA),
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA)
    };

    std::cout << "=== Backward Test ===" << std::endl;
    for (int i = 0; i < 4; i++) {
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
                auto [flash_dq, flash_dk, flash_dv] = V2::backward_launch(
                    params[i], q.detach(), k.detach(), v.detach(), flash_out, flash_lse, grad_o
                );
            };
        }
    }
}

void test_function() {
    std::cout << "=== Autograd Function Test ===" << std::endl;
    {
        Params params{.batch_size=4, .num_heads=8, .seq_len=512, .head_dim=64};
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

        // Create base tensors, then clone for independent autograd paths
        torch::Tensor q_base = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k_base = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v_base = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

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
        std::cout << "  dQ: " << (q_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  dK: " << (k_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  dV: " << (v_ok ? "PASS" : "FAIL") << std::endl;
    }
}

int main() {
    test_forward();
    test_backward();
    test_function();
    return 0;
}
