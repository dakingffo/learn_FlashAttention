#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    template <size_t Num>
    CUTE_HOST_DEVICE constexpr size_t log2() noexcept {
        static_assert(!(Num & (Num - 1)));
        return cute::log_2(Num);
    }
}