#include "torch_utils.hpp"

namespace FlashAttention::utility {
    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k) {
        return torch::matmul(q, k.transpose(-2, -1));
    }

    torch::Tensor standard_matmul(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
        torch::Tensor s = torch::matmul(q, k.transpose(-2, -1));
        return torch::matmul(s, v);
    }

    torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
        torch::Tensor s = torch::matmul(q, k.transpose(-2, -1));
        s = s / std::sqrt(q.size(-1));
        torch::Tensor p = torch::softmax(s, /*dim=*/-1);
        return torch::matmul(p, v);
    }

    std::tuple<torch::Tensor, torch::Tensor> naive_attention_with_LSE(
        torch::Tensor q, torch::Tensor k, torch::Tensor v
    ) {
        int64_t batch_size = q.size(0);
        int64_t num_heads  = q.size(1);
        int64_t seq_len    = q.size(2);
        int64_t head_dim   = q.size(3);
        
        double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

        auto out_options = q.options();
        torch::Tensor total_out = torch::empty({batch_size, num_heads, seq_len, head_dim}, out_options);
        torch::Tensor total_lse = torch::empty({batch_size, num_heads, seq_len}, q.options().dtype(torch::kFloat32));

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t h = 0; h < num_heads; ++h) {
                torch::Tensor q_sub = q.select(0, b).select(0, h);
                torch::Tensor k_sub = k.select(0, b).select(0, h);
                torch::Tensor v_sub = v.select(0, b).select(0, h);

                torch::Tensor s_sub = torch::matmul(q_sub, k_sub.transpose(-2, -1)) * scale;

                torch::Tensor s_sub_f32 = s_sub.to(torch::kFloat32);
                
                auto max_results = torch::max(s_sub_f32, /*dim=*/-1);
                torch::Tensor m_sub = std::get<0>(max_results); 

                torch::Tensor exp_shifted = torch::exp(s_sub_f32 - m_sub.unsqueeze(-1));
                torch::Tensor sum_exp = torch::sum(exp_shifted, /*dim=*/-1); // [seq_len]
                torch::Tensor log_sum_exp = torch::log(sum_exp);             // [seq_len]
                
                torch::Tensor lse_sub = m_sub + log_sum_exp; // [seq_len]

                torch::Tensor p_sub = torch::softmax(s_sub_f32, /*dim=*/-1).to(q.scalar_type()); 
                torch::Tensor o_sub = torch::matmul(p_sub, v_sub); // [seq_len, head_dim]

                total_out.select(0, b).select(0, h).copy_(o_sub);
                total_lse.select(0, b).select(0, h).copy_(lse_sub);
            }
        }

        return std::make_tuple(total_out, total_lse);
    }

    bool check_equal(torch::Tensor t1, torch::Tensor t2, float tol, int print_count) {
        auto t1_view = t1.view(-1);
        auto t2_view = t2.view(-1);

        std::cout << "First " << print_count << ":\n";
        std::cout << "t1: ";

        for (int i = 0; i < print_count; i++) {
            std::cout << t1_view[i].item<float>() << " \n"[i == print_count - 1];
        }
        
        std::cout << "t2: ";
        for (int i = 0; i < print_count; i++) {
            std::cout << t2_view[i].item<float>() << " \n"[i == print_count - 1];
        }
        if (torch::isnan(t1).any().item<bool>() || torch::isnan(t2).any().item<bool>()) {
            std::cout << "Error: NaN detected in tensors!" << std::endl;
            return false;
        }
        torch::Tensor mask = torch::abs(t1_view - t2_view) > tol;
        if (!mask.any().item<bool>()) {
            std::cout << "All close!" << std::endl;
            return true;
        }
        torch::Tensor indices = torch::nonzero(mask);

        std::cout << "Not equal: " << indices.size(0) << std::endl;

        for (int i = 0; i < indices.size(0); ++i) {
            int idx = indices[i].item<int>();

            float val1 = t1_view[idx].item<float>();
            float val2 = t2_view[idx].item<float>();

            printf("[%d] -> t1: %f , but t2: %f \n", idx, val1, val2);
            
            if (i >= print_count) {
                std::cout << "..." << std::endl;
                break;
            }
        }

        return false;
    }
}