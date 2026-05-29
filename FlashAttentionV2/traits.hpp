#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include <utility/math.hpp>
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
        size_t Dim, size_t Pipe, size_t TileParallel, size_t TileSequence, size_t EUReapt,
        typename MMAOP /* = SM80_16x8x16_F32BF16BF16F32_TN */
    >
    struct Traits {
        static_assert(Dim >= TileSequence,
            "HeadDim should be no less than TileSequence for register reuse.");
        static_assert(Pipe >= 2);

        using mma_op         = MMAOP;
        using mma_traits     = MMA_Traits<mma_op>;
        using mma_atom       = MMA_Atom<mma_traits>;
        using mma_atom_shape = typename mma_traits::Shape_MNK;
        static_assert(std::is_same_v<typename mma_traits::ValTypeA, typename mma_traits::ValTypeB>);
        static_assert(std::is_same_v<typename mma_traits::ValTypeD, typename mma_traits::ValTypeC>);

        using type              = typename mma_traits::ValTypeA;
        using acc_type          = typename mma_traits::ValTypeD;
        using lse_type          = float;
        using torch_type        = utility::torch_type_of_t<type>;
        using pointer           = type*;
        using const_pointer     = const type*;
        using lse_pointer       = float*;
        using const_lse_pointer = const float*;

        static constexpr int SwizzleBase  = utility::log2<16 / sizeof(type)>();
        static constexpr int SwizzleShift = 3;
        template <size_t SmemColAtom>
        static constexpr int SwizzleBits  = utility::log2<8 * SmemColAtom>() - SwizzleShift - SwizzleBase;
        template <size_t SmemColAtom>
        using SwizzleOf = Swizzle<SwizzleBits<SmemColAtom>, SwizzleBase, SwizzleShift>;

        struct QParallel {
            static constexpr int HeadDim = Dim;
            static constexpr int TileQ   = TileParallel;
            static constexpr int TileKV  = TileSequence;
            static constexpr int Stage   = Pipe;

            using type              = typename Traits::type;
            using acc_type          = typename Traits::acc_type;
            using lse_type          = typename Traits::lse_type;
            using torch_type        = typename Traits::torch_type;
            using pointer           = typename Traits::pointer;
            using const_pointer     = typename Traits::const_pointer;
            using lse_pointer       = typename Traits::lse_pointer;
            using const_lse_pointer = typename Traits::const_lse_pointer;

            static constexpr int AtomM     = get<0>(mma_atom_shape{});
            static constexpr int AtomN     = get<1>(mma_atom_shape{});
            static constexpr int AtomK     = get<2>(mma_atom_shape{}); // only one warp
            static constexpr int EURepeatM = EUReapt;
            static constexpr int PermuteN  = get<2>(mma_atom_shape{}) / get<1>(mma_atom_shape{});
            static constexpr int PM        = EURepeatM * AtomM;
            static constexpr int PN        = PermuteN * AtomN;
            static constexpr int PK        = AtomK; // only one warp
            static_assert(TileQ >= PM && TileKV >= PN);

            using CLayout = typename mma_traits::CLayout;
            using MMA     = decltype(make_tiled_mma(
                mma_op{},
                make_layout(Shape<Int<EURepeatM>, _1, _1>{}),
                Tile<Int<PM>, Int<PN>, Int<PK>>{}
            ));

            static constexpr int ThreadsPerRow  = size<0, 0>(CLayout{});
            static constexpr int RowsPerThread  = size<1, 1>(CLayout{}) * TileQ / PM;
            static constexpr int ThreadsPerCol  = size<0, 1>(CLayout{});
            static constexpr int ThreadsPerAtom = ThreadsPerRow * ThreadsPerCol;
            static constexpr int ThreadsPerCTA  = thr_size(MMA{});

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

            static constexpr int SmemHeadAtom = HeadDim % 64 == 0 ? 64 : 32;
            using SmemLayoutQOLogical  = decltype(tile_to_shape(
                make_layout(Shape<_8, Int<SmemHeadAtom>>{}, Stride<Int<SmemHeadAtom>, _1>{}),
                Shape<Int<TileQ>, Int<HeadDim>>{}
            ));
            using SmemLayoutQOTLogical = decltype(select<1, 0>(SmemLayoutQOLogical{}));
            using SmemLayoutKVLogical  = decltype(tile_to_shape(
                make_layout(Shape<_8, Int<SmemHeadAtom>>{}, Stride<Int<SmemHeadAtom>, _1>{}),
                Shape<Int<TileKV>, Int<HeadDim>, Int<Stage>>{}
            ));
            using SmemLayoutKVTLogical = decltype(select<1, 0, 2>(SmemLayoutKVLogical{}));
            
            using SmemLayoutQO  = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutQOLogical{}));
            using SmemLayoutQOT = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutQOTLogical{}));
            using SmemLayoutKV  = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutKVLogical{}));
            using SmemLayoutKVT = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutKVTLogical{}));
        };

        struct KVParallel {
            static constexpr int HeadDim = Dim;
            static constexpr int TileQ   = TileSequence;
            static constexpr int TileKV  = TileParallel;
            static constexpr int Stage   = Pipe;

            using type              = typename Traits::type;
            using acc_type          = typename Traits::acc_type;
            using lse_type          = typename Traits::lse_type;
            using torch_type        = typename Traits::torch_type;
            using pointer           = typename Traits::pointer;
            using const_pointer     = typename Traits::const_pointer;
            using lse_pointer       = typename Traits::lse_pointer;
            using const_lse_pointer = typename Traits::const_lse_pointer;

            static constexpr int AtomM     = get<0>(mma_atom_shape{});
            static constexpr int AtomN     = get<1>(mma_atom_shape{});
            static constexpr int AtomK     = get<2>(mma_atom_shape{}); // only one warp
            static constexpr int EURepeatN = 1; // maybe optimized in future
            static constexpr int PermuteN  = get<2>(mma_atom_shape{}) / get<1>(mma_atom_shape{});
            static constexpr int PM        = AtomM;
            static constexpr int PN        = PermuteN * AtomN;
            static constexpr int PK        = AtomK; // only one warp
            static_assert(TileQ >= PM && TileKV >= PN);

            using CLayout = typename mma_traits::CLayout;
            using MMA     = decltype(make_tiled_mma(
                mma_op{},
                make_layout(Shape<_1, Int<EURepeatN>, _1>{}),
                Tile<Int<PM>, Int<PN>, Int<PK>>{}
            ));

            static constexpr int ThreadsPerRow  = size<0, 0>(CLayout{});
            static constexpr int RowsPerThread  = size<1, 1>(CLayout{}) * TileQ / PM;
            static constexpr int ThreadsPerCol  = size<0, 1>(CLayout{});
            static constexpr int ThreadsPerAtom = ThreadsPerRow * ThreadsPerCol;
            static constexpr int ThreadsPerCTA  = thr_size(MMA{});

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
            using S2RCopyAT       = decltype(make_tiled_copy_A(
                s2r_copy_atom_T{},
                MMA{}
            ));
            using S2RCopyBT       = decltype(make_tiled_copy_B(
                s2r_copy_atom_T{},
                MMA{}
            ));

            using r2s_copy_atom   = Copy_Atom<UniversalCopy<int>, type>;
            using R2SCopyA        = decltype(make_tiled_copy_A(
                r2s_copy_atom{},
                MMA{}
            ));
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

            static constexpr int SmemHeadAtom = HeadDim % 64 == 0 ? 64 : 32;
            using SmemLayoutQOLogical  = decltype(tile_to_shape(
                make_layout(Shape<_8, Int<SmemHeadAtom>>{}, Stride<Int<SmemHeadAtom>, _1>{}),
                Shape<Int<TileQ>, Int<HeadDim>, Int<Stage>>{}
            ));
            using SmemLayoutQOTLogical = decltype(select<1, 0, 2>(SmemLayoutQOLogical{}));
            using SmemLayoutKVLogical  = decltype(tile_to_shape(
                make_layout(Shape<_8, Int<SmemHeadAtom>>{}, Stride<Int<SmemHeadAtom>, _1>{}),
                Shape<Int<TileKV>, Int<HeadDim>>{}
            ));
            using SmemLayoutKVTLogical = decltype(select<1, 0>(SmemLayoutKVLogical{}));
            
            static constexpr int SmemTileAtom = TileKV % 64 == 0 ? 64 : 32;
            using SmemLayoutSPLogical  = decltype(tile_to_shape(
                make_layout(Shape<_8, Int<SmemTileAtom>>{}, Stride<Int<SmemTileAtom>, _1>{}),
                Shape<Int<TileQ>, Int<TileKV>>{}
            ));
            using SmemLayoutSPTLogical = decltype(select<1, 0>(SmemLayoutSPLogical{}));
            
            using SmemLayoutQO  = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutQOLogical{}));
            using SmemLayoutQOT = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutQOTLogical{}));
            using SmemLayoutKV  = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutKVLogical{}));
            using SmemLayoutKVT = decltype(composition(SwizzleOf<SmemHeadAtom>{}, SmemLayoutKVTLogical{}));
            using SmemLayoutSP  = decltype(composition(SwizzleOf<SmemTileAtom>{}, SmemLayoutSPLogical{}));
            using SmemLayoutSPT = decltype(composition(SwizzleOf<SmemTileAtom>{}, SmemLayoutSPTLogical{}));
        };
    };

    template <typename CutlassType, size_t HeadDim> 
    struct DispatchTraits;

    template <>
    struct DispatchTraits<cutlass::half_t, 32> {
        static constexpr size_t HeadDim      = 32;
        static constexpr size_t Pipe         = 3;
        static constexpr size_t TileParallel = 128;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 8;
        using MMAOP      = SM80_16x8x16_F32F16F16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };

    template <>
    struct DispatchTraits<cutlass::half_t, 64> {
        static constexpr size_t HeadDim      = 64;
        static constexpr size_t Pipe         = 3;
        static constexpr size_t TileParallel = 64;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 4;
        using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };

    template <>
    struct DispatchTraits<cutlass::half_t, 128> {
        static constexpr size_t HeadDim      = 128;
        static constexpr size_t Pipe         = 2;
        static constexpr size_t TileParallel = 64;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 4;
        using MMAOP = SM80_16x8x16_F32F16F16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 32> {
        static constexpr size_t HeadDim      = 32;
        static constexpr size_t Pipe         = 3;
        static constexpr size_t TileParallel = 128;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 8;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 64> {
        static constexpr size_t HeadDim      = 64;
        static constexpr size_t Pipe         = 3;
        static constexpr size_t TileParallel = 64;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 4;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };

    template <>
    struct DispatchTraits<cutlass::bfloat16_t, 128> {
        static constexpr size_t HeadDim      = 128;
        static constexpr size_t Pipe         = 2;
        static constexpr size_t TileParallel = 64;
        static constexpr size_t TileSequence = 32;
        static constexpr size_t EUReapt      = 4;
        using MMAOP = SM80_16x8x16_F32BF16BF16F32_TN;
        using QParallel  = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::QParallel;
        using KVParallel = typename Traits<HeadDim, Pipe, TileParallel, TileSequence, EUReapt, MMAOP>::KVParallel;
    };
}