#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include <utility/type_traits.hpp>

namespace FlashAttention::V2 {
    using namespace cute;
    using namespace cutlass;

    template <size_t Bytes>
    struct DispatchS2RCopy;

    template <>
    struct DispatchS2RCopy<2> {
        template <typename Ty>
        using copy_atom   = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, Ty>;
        template <typename Ty>
        using copy_atom_T = Copy_Atom<Copy_Traits<SM75_U16x8_LDSM_T>, Ty>;
    };

    template <
        size_t   Pipe    = 5,
        size_t   QPerCTA = 64,
        size_t   TileSeq = 64,
        size_t   EUReapt = 4,
        typename MMAOP   = SM80_16x8x16_F32F16F16F32_TN
    >
    struct Traits {
        static constexpr int TileQ   = QPerCTA;
        static constexpr int TileKV  = TileSeq;
        static constexpr int Stage   = Pipe;
        static_assert(Stage >= 2);

        using mma_op         = MMAOP;
        using mma_traits     = MMA_Traits<mma_op>;
        using mma_atom       = MMA_Atom<mma_traits>;
        using mma_atom_shape = typename mma_traits::Shape_MNK;

        static_assert(std::is_same_v<typename mma_traits::ValTypeA, typename mma_traits::ValTypeB>);
        static_assert(std::is_same_v<typename mma_traits::ValTypeD, typename mma_traits::ValTypeC>);
        using type          = typename mma_traits::ValTypeA;
        using acc_type      = typename mma_traits::ValTypeD;
        using lse_type      = float;
        using torch_type    = utility::torch_type_of_t<type>;
        using pointer       = type*;
        using const_pointer = const type*;
        using lse_pointer   = float*;

        static constexpr int AtomM     = get<0>(mma_atom_shape{});
        static constexpr int AtomN     = get<1>(mma_atom_shape{});
        static constexpr int AtomK     = get<2>(mma_atom_shape{}); // only one warp
        static constexpr int EURepeatM = EUReapt;
        static constexpr int PermuteN  = get<2>(mma_atom_shape{}) / get<1>(mma_atom_shape{});
        static constexpr int PM        = EURepeatM * AtomM;
        static constexpr int PN        = PermuteN * AtomN;
        static constexpr int PK        = AtomK; // only one warp
        static_assert(TileQ >= PM && TileKV >= PN);

        using MMA = decltype(make_tiled_mma(
            mma_op{},
            make_layout(Shape<Int<EURepeatM>, _1, _1>{}),
            Tile<Int<PM>, Int<PN>, Int<PK>>{}
        ));

        static constexpr int ThreadsPerCTA = thr_size(MMA{});

        using CopyThreadsLayout = decltype(make_layout(Shape<Int<ThreadsPerCTA / 4>, _4>{}, Stride<_4, _1>{}));

        using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, type>;
        using G2SCopy        = decltype(make_tiled_copy(
            g2s_copy_atom{},
            CopyThreadsLayout{},
            make_layout(Shape<_1, Int<sizeof(uint128_t) / sizeof(type)>>{})
        ));

        using s2r_copy_atom   = typename DispatchS2RCopy<sizeof(type)>::template copy_atom<type>;
        using S2RCopyA        = decltype(make_tiled_copy_A(
            s2r_copy_atom{},
            MMA{}
        ));
        using S2RCopyB        = decltype(make_tiled_copy_B(
            s2r_copy_atom{},
            MMA{}
        ));

        using s2r_copy_atom_T = typename DispatchS2RCopy<sizeof(type)>::template copy_atom_T<type>;
        using S2RCopyBT       = decltype(make_tiled_copy_B(
            s2r_copy_atom_T{},
            MMA{}
        ));

        using r2s_copy_atom   = Copy_Atom<UniversalCopy<int>, type>;
        using R2SCopyC        = decltype(make_tiled_copy_C(
            r2s_copy_atom{},
            MMA{}
        ));

        using s2g_copy_atom   = Copy_Atom<UniversalCopy<uint128_t>, type>;
        using S2GCopy         = decltype(make_tiled_copy(
            s2g_copy_atom{},
            CopyThreadsLayout{},
            make_layout(Shape<_1, Int<sizeof(uint128_t) / sizeof(type)>>{})
        ));
    };

    template <typename T, int HeadDim> 
    struct DispatchTraits;

    template <>
    struct DispatchTraits<cutlass::half_t, 32> {
        static constexpr int Pipe    = 5;
        static constexpr int QPerCTA = 128;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 8;
        using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };

    template <>
    struct DispatchTraits<cutlass::half_t, 64> {
        static constexpr int Pipe    = 5;
        static constexpr int QPerCTA = 64;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 4;
        using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };

    template <>
    struct DispatchTraits<cutlass::half_t, 128> {
        static constexpr int Pipe    = 2;
        static constexpr int QPerCTA = 64;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 4;
        using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 32> {
        static constexpr int Pipe    = 5;
        static constexpr int QPerCTA = 128;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 8;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 64> {
        static constexpr int Pipe    = 5;
        static constexpr int QPerCTA = 64;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 4;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 128> {
        static constexpr int Pipe    = 2;
        static constexpr int QPerCTA = 64;
        static constexpr int TileSeq = 64;
        static constexpr int EUReapt = 4;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using type  = Traits<Pipe, QPerCTA, TileSeq, EUReapt, MMAOP>;
    };
}