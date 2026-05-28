#pragma once
#include <cute/tensor.hpp>
#include <type_traits>

namespace FlashAttention::utility {
    template <
        size_t Dim, size_t CTATile, bool FillOutOfBoundary = true,
        typename TiledCopy, typename Identity,
        typename SrcTensor, typename DstTensor
    >
    CUTE_DEVICE void copy_within_boundary(
        TiledCopy&& tiled_copy, 
        signed long len, unsigned int idx, 
        const Identity& identity,
        const SrcTensor& src, DstTensor&& dst,
        typename std::decay_t<DstTensor>::value_type val = typename std::decay_t<DstTensor>::value_type{0.0}
    ) {
        using namespace cute;

        if ((idx + 1) * CTATile - 1 < len) {
            copy(tiled_copy, src, dst);
        }
        else {
            auto mask = make_tensor<bool>(shape(identity));
            CUTE_UNROLL
            for (int i = 0; i < size(identity); i++) {
                bool within_boundary = (idx * CTATile + get<Dim>(identity(i)) < len);
                mask(i) = within_boundary;
                if constexpr (FillOutOfBoundary) {
                    dst(i) = (within_boundary ? dst(i) : val);
                }
            }
            copy_if(tiled_copy, mask, src, dst);
        }
    }

    template <size_t Dim, size_t CTATile, typename Tensor, typename Identity>
    CUTE_DEVICE void fill_cross_boundary(
        Tensor&& tensor, 
        signed long len, unsigned int idx,
        const Identity& identity,
        typename std::decay_t<Tensor>::value_type internal,
        typename std::decay_t<Tensor>::value_type external
    ) { 
        using namespace cute;

        if ((idx + 1) * CTATile - 1 < len) {
            fill(tensor, internal);
        }
        else {
            CUTE_UNROLL
            for (int i = 0; i < size(identity); i++) {
                tensor(i) = (idx * CTATile + get<Dim>(identity(i)) < len ? internal : external);
            }
        }
    }
}