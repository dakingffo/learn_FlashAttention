#pragma once
#include <torch/torch.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

namespace FlashAttention::utility {
    template <typename CutlassType>
    struct torch_type_of;

    template <>
    struct torch_type_of<cutlass::half_t>{
        using type = c10::Half;
        static constexpr c10::ScalarType value = c10::kHalf;
    };
    
    template <>
    struct torch_type_of<cutlass::bfloat16_t>{
        using type = c10::BFloat16;
        static constexpr c10::ScalarType value = c10::kBFloat16;
    };

    template <typename CutlassType>
    using torch_type_of_t = typename torch_type_of<CutlassType>::type;

    template <typename CutlassType>
    static constexpr c10::ScalarType torch_type_of_v = torch_type_of<CutlassType>::value;
}