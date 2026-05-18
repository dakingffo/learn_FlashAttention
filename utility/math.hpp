#pragma once
#include <cute/tensor.hpp>

namespace FlashAttention::utility {
    template <size_t Num>
    CUTE_HOST_DEVICE constexpr size_t log2() noexcept {
        static_assert(!(Num & (Num - 1)));
        return cute::log_2(Num);
    }

    template <size_t Num>
    CUTE_HOST_DEVICE constexpr float sqrt() noexcept {
        static_assert(Num == 32 || Num == 64 || Num == 128 || Num == 256);
        switch (Num){
        case 32:  return 5.65685;
        case 64:  return 8.0;
        case 128: return 11.31370;
        case 256: return 16.0;
        }
        __builtin_unreachable();
    }
}