#pragma once
#include <torch/torch.h>

#define DISPATCH_VALUE(TYPE, NAME, VAL, ...)                                \
{                                                                           \
    switch (VAL) {                                                          \
    /*case 32  : {static constexpr TYPE NAME = 32;  __VA_ARGS__; break; } */ \
    case 64  : { static constexpr TYPE NAME = 64;  __VA_ARGS__; break; }    \
    case 128 : { static constexpr TYPE NAME = 128; __VA_ARGS__; break; }    \
    default  : C10_THROW_ERROR(ValueError, "Unsupported value");            \
    }                                                                       \
}

#define DISPATCH_TYPE(TYPENAME, SCALAR_TYPE, ...)                                       \
{                                                                                       \
    switch (SCALAR_TYPE) {                                                              \
    case c10::kHalf     : { using TYPENAME = cutlass::half_t;     __VA_ARGS__; break; } \
    case c10::kBFloat16 : { using TYPENAME = cutlass::bfloat16_t; __VA_ARGS__; break; } \
    default  : C10_THROW_ERROR(ValueError, "Unsupported value");                        \
    }                                                                                   \
}
