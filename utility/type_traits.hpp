#pragma once
#include <torch/torch.h>
#include <cutlass/numeric_types.h>
#include <type_traits>

namespace FlashAttention::utility {
    template <typename CutlassType>
    struct torch_type_of;

    template <>
    struct torch_type_of<cutlass::half_t>{
        using type = at::Half;
    };
    
    template <>
    struct torch_type_of<cutlass::bfloat16_t>{
        using type = at::BFloat16;
    };

    template <>
    struct torch_type_of<float>{
        using type = float;
    };

    template <>
    struct torch_type_of<double>{
        using type = double;
    };

    template <typename CutlassType>
    using torch_type_of_t = typename torch_type_of<CutlassType>::type;
}