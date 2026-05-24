#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    struct Params {
        int64_t batch_size;
        int64_t num_heads;
        int64_t seq_len;
        int64_t head_dim;
        float   scale = 0;

        CUTE_HOST_DEVICE signed long one_head_size() const noexcept {
            return seq_len * head_dim;
        }

        CUTE_HOST_DEVICE signed long one_batch_size() const noexcept {
            return num_heads * one_head_size();
        }
    };
}