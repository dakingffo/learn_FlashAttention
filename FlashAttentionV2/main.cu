#include "flash_attention_v2.hpp"
#include <utility/timer.hpp>

int main() {
    using namespace FlashAttention;
    using namespace FlashAttention::utility;
    {
        std::cout << "Params<HeadDim=64>{.batch_size=8, .num_heads=64, .seq_len=2048}:" << std::endl;
        Parameters<64> params{.batch_size=8, .num_heads=64, .seq_len=2048};
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

        torch::Tensor q = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        torch::Tensor naive_out = naive_attention(q, k, v);
        torch::Tensor flash_out = V2::forward_launch<V2::Traits<5>>(params, q, k, v); // 5 Pipeline

        if (utility::check_equal(naive_out, flash_out)) {
            TIMING("NaiveAttention", config<loop<5>>) {
                naive_out = naive_attention(q, k, v);
            };
            TIMING("FlashAttentionV2", config<loop<5>>) {
                flash_out = V2::forward_launch<V2::Traits<5>>(params, q, k, v);
            };
        }
    }
    {
        std::cout << "Params<HeadDim=128>{.batch_size=8, .num_heads=64, .seq_len=1000}:" << std::endl;
        Parameters<128> params{.batch_size=8, .num_heads=64, .seq_len=1000};
        auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

        torch::Tensor q = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor k = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);
        torch::Tensor v = torch::randn({params.batch_size, params.num_heads, params.seq_len, params.head_dim}, options);

        torch::Tensor naive_out = naive_attention(q, k, v);
        torch::Tensor flash_out = V2::forward_launch<V2::Traits<2>>(params, q, k, v); // 2 Pipeline

        if (utility::check_equal(naive_out, flash_out)) {
            TIMING("NaiveAttention", config<loop<5>>) {
                naive_out = naive_attention(q, k, v);
            };
            TIMING("FlashAttentionV2", config<loop<5>>) {
                flash_out = V2::forward_launch<V2::Traits<2>>(params, q, k, v);
            };
        }
    }

    return 0;
}