#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    template <size_t HeadDim = 64>
    struct Parameters {
        signed long batch_size;
        signed long num_heads;
        signed long seq_len;
        static constexpr signed long head_dim = HeadDim;

        CUTE_HOST_DEVICE signed long one_head_size() const noexcept {
            return seq_len * head_dim;
        }

        CUTE_HOST_DEVICE signed long one_batch_size() const noexcept {
            return num_heads * one_head_size();
        }
    };
}