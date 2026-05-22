#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    struct Params {
        int   batch_size;
        int   num_heads;
        int   seq_len;
        int   head_dim;
        float scale = 0;

        CUTE_HOST_DEVICE signed long one_head_size() const noexcept {
            return seq_len * head_dim;
        }

        CUTE_HOST_DEVICE signed long one_batch_size() const noexcept {
            return num_heads * one_head_size();
        }
    };
}