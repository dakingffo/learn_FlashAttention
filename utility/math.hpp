#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    template <size_t Num>
    CUTE_HOST_DEVICE constexpr size_t log2() noexcept {
        static_assert(!(Num & (Num - 1)));
        return cute::log_2(Num);
    }

    template <size_t Num>
    CUTE_HOST_DEVICE constexpr size_t sqrt() noexcept {
        static_assert(!(Num & (Num - 1)));
        constexpr int n = log2<Num>();
        static_assert(n % 2 == 0);
        return Num >> (n / 2);
    }
}