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

    bool check_equal(torch::Tensor t1, torch::Tensor t2, float tol, int print_count) {
        using at::indexing::Slice;
        if (torch::isnan(t1).any().item<bool>() || torch::isnan(t2).any().item<bool>()) {
            std::cout << "Error: NaN detected in tensors!" << std::endl;
            return false;
        }
        
        std::cout << "First " << print_count << ":\n";
        std::cout << "t1: ";
        for (int i = 0; i < print_count; i++) {
            std::cout << t1[0][0][0][i].item<float>() << " \n"[i == print_count - 1];
        }
        std::cout << "t2: ";
        for (int i = 0; i < print_count; i++) {
            std::cout << t2[0][0][0][i].item<float>() << " \n"[i == print_count - 1];
        }

        torch::Tensor mask = torch::abs(t1 - t2) > tol;
        if (!mask.any().item<bool>()) {
            std::cout << "All close!" << std::endl;
            return true;
        }
        torch::Tensor indices = torch::nonzero(mask);

        std::cout << "Not equal: " << indices.size(0) << std::endl;

        for (int i = 0; i < indices.size(0); ++i) {
            int idx_i = indices[i][0].item<int>();
            int idx_j = indices[i][1].item<int>();
            int idx_k = indices[i][2].item<int>();
            int idx_l = indices[i][3].item<int>();

            float val1 = t1[idx_i][idx_j][idx_k][idx_l].item<float>();
            float val2 = t2[idx_i][idx_j][idx_k][idx_l].item<float>();

            printf("[%d, %d, %d, %d] ->", idx_i, idx_j, idx_k, idx_l);
            std::cout << " t1: " << val1 << " t2: " << val2 << std::endl;
            
            if (i > print_count) {
                std::cout << "..." << std::endl;
                break;
            }
        }

        return false;
    }
}