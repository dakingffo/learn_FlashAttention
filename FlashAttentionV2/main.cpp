#include <utility/timer.hpp>
#include <utility/torch_utils.hpp>

#include "flash_attention_v2.hpp"

int main() {
    using namespace FlashAttention;
    using namespace FlashAttention::utility;
    {
        std::cout << "[FP16] Params{.batch_size=8, .num_heads=64, .seq_len=2048, .head_dim=64}:" << std::endl;
        Params params{.batch_size=8, .num_heads=64, .seq_len=2048, .head_dim=64};
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

        torch::Tensor q = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        auto [naive_out, naive_lse] = naive_attention_with_LSE(q, k, v);
        auto [flash_out, flash_lse] = V2::forward_launch(params, q, k, v);

        if (utility::check_equal(naive_out, flash_out) && utility::check_equal(naive_lse, flash_lse)) {
            // TIMING("NaiveAttention", config<loop<5>>) {
            //     naive_out = naive_attention(q, k, v);
            // };
            TIMING("FlashAttentionV2", config<loop<5>>) {
                flash_out = std::get<0>(V2::forward_launch(params, q, k, v));
            };
        }
    }
    /*
    {
        std::cout << "[FP16] Params{.batch_size=8, .num_heads=64, .seq_len=2000, .head_dim=128}:" << std::endl;
        Params params{.batch_size=8, .num_heads=64, .seq_len=2000, .head_dim=128};
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

        torch::Tensor q = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        auto [naive_out, naive_lse] = naive_attention_with_LSE(q, k, v);
        auto [flash_out, flash_lse] = V2::forward_launch(params, q, k, v);

        if (utility::check_equal(naive_out, flash_out) && utility::check_equal(naive_lse, flash_lse)) {
            // TIMING("NaiveAttention", config<loop<5>>) {
            //     naive_out = naive_attention(q, k, v);
            // };
            TIMING("FlashAttentionV2", config<loop<5>>) {
                flash_out = std::get<0>(V2::forward_launch(params, q, k, v));
            };
        }
    }
    {
        std::cout << "[BF16] Params{.batch_size=8, .num_heads=64, .seq_len=2000, .head_dim=64}:" << std::endl;
        Params params{.batch_size=8, .num_heads=64, .seq_len=2000, .head_dim=64};
        auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);

        torch::Tensor q = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        auto [naive_out, naive_lse] = naive_attention_with_LSE(q, k, v);
        auto [flash_out, flash_lse] = V2::forward_launch(params, q, k, v);

        if (utility::check_equal(naive_out, flash_out) && utility::check_equal(naive_lse, flash_lse)) {
            // TIMING("NaiveAttention", config<loop<5>>) {
            //     naive_out = naive_attention(q, k, v);
            // };
            TIMING("FlashAttentionV2", config<loop<5>>) {
                flash_out = std::get<0>(V2::forward_launch(params, q, k, v));
            };
        }
    }*/
    return 0;
}
