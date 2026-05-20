#ifndef AXN_PRTN_SIMD_SSE_I32_H
#define AXN_PRTN_SIMD_SSE_I32_H

/**
 * @file SIMD/SSE/i32.h
 * @brief SSE implementation for 32-bit integer SIMD (`Proton::Simd<4,i32>`).
 *
 * Uses SSE intrinsics for integer arithmetic, bitwise ops and shifts.
 */

#include <math.h> // IWYU pragma: keep

#include "../Simd.h"
#include <Quark/System/Simd.h> // IWYU pragma: keep
#include <Quark/Types.h>

#undef ATOM_SIMD_SSE2

#if ATOM_SIMD_SSE41
#    include <smmintrin.h>
#endif
#if ATOM_SIMD_SSSE3
#    include <tmmintrin.h>
#endif
#if ATOM_SIMD_SSE2
#    include <emmintrin.h>
#endif
#include <xmmintrin.h>

namespace Proton {
    template <>
    union Simd<4, i32> {
#if ATOM_SIMD_SSE2
        __m128i v128;
#else
        __m128 v128;
#endif
        i32 i[4];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128i a0) { v128 = a0; }
#else
        Simd(__m128 a0) { v128 = a0; }
#endif
        Simd(i32 a0, i32 a1, i32 a2, i32 a3) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
        }

        static inline Simd<4, i32> load(const i32* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_loadu_si128((const __m128i*)ptr));
#else
            return Simd<4, i32>(_mm_loadu_ps((const f32*)ptr));
#endif
        }
        static inline void store(i32* ptr, const Simd<4, i32> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_si128((__m128i*)ptr, simd.v128);
#else
            _mm_storeu_ps((f32*)ptr, simd.v128);
#endif
        }
        static inline Simd<4, i32> set1(const i32 value) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_set1_epi32(value));
#else
            return Simd<4, i32>(_mm_set1_ps(*(f32*)&value));
#endif
        }
        static inline Simd<4, i32> set(const i32 a0, const i32 a1, const i32 a2, const i32 a3) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_set_epi32(a3, a2, a1, a0));
#else
            return Simd<4, i32>(_mm_set_ps(*(f32*)&a3, *(f32*)&a2, *(f32*)&a1, *(f32*)&a0));
#endif
        }

        static inline Simd<4, i32> add(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_add_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(a.i[0] + b.i[0], a.i[1] + b.i[1], a.i[2] + b.i[2], a.i[3] + b.i[3]);
#endif
        }
        static inline Simd<4, i32> sub(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_sub_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(a.i[0] - b.i[0], a.i[1] - b.i[1], a.i[2] - b.i[2], a.i[3] - b.i[3]);
#endif
        }
        static inline Simd<4, i32> abs(const Simd<4, i32> &a) {
#if ATOM_SIMD_SSSE3
            return Simd<4, i32>(_mm_abs_epi32(a.v128));
#elif ATOM_SIMD_SSE2
            __m128i sign = _mm_srai_epi32(a.v128, 31);
            return Simd<4, i32>(_mm_sub_epi32(_mm_xor_si128(a.v128, sign), sign));
#else
            return Simd<4, i32>(labs(a.i[0]), labs(a.i[1]), labs(a.i[2]), labs(a.i[3]));
#endif
        }
        static inline Simd<4, i32> neg(const Simd<4, i32> &a) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_sub_epi32(_mm_setzero_si128(), a.v128));
#else
            return Simd<4, i32>(-a.i[0], -a.i[1], -a.i[2], -a.i[3]);
#endif
        }
        static inline Simd<4, i32> min(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<4, i32>(_mm_min_epi32(a.v128, b.v128));
#elif ATOM_SIMD_SSE2
            __m128i mask = _mm_cmplt_epi32(a.v128, b.v128);
            return Simd<4, i32>(
                _mm_or_si128(_mm_and_si128(mask, a.v128), _mm_andnot_si128(mask, b.v128)));
#else
            return Simd<4, i32>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3]);
#endif
        }
        static inline Simd<4, i32> max(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<4, i32>(_mm_max_epi32(a.v128, b.v128));
#elif ATOM_SIMD_SSE2
            __m128i mask = _mm_cmpgt_epi32(a.v128, b.v128);
            return Simd<4, i32>(
                _mm_or_si128(_mm_and_si128(mask, a.v128), _mm_andnot_si128(mask, b.v128)));
#else
            return Simd<4, i32>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3]);
#endif
        }

        static inline Simd<4, i32> land(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_and_si128(a.v128, b.v128));
#else
            return Simd<4, i32>(_mm_and_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<4, i32> lor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_or_si128(a.v128, b.v128));
#else
            return Simd<4, i32>(_mm_or_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<4, i32> lxor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_xor_si128(a.v128, b.v128));
#else
            return Simd<4, i32>(_mm_xor_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<4, i32> lnot(const Simd<4, i32> &a) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_xor_si128(a.v128, _mm_set1_epi32(~0)));
#else
            return Simd<4, i32>(_mm_xor_ps(a.v128, _mm_set_ps1(~0)));
#endif
        }
        static inline Simd<4, i32> landnot(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_andnot_si128(b.v128, a.v128));
#else
            return Simd<4, i32>(_mm_andnot_ps(b.v128, a.v128));
#endif
        }

        static inline Simd<4, i32> shl(const Simd<4, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_slli_epi32(a.v128, count));
#else
            return Simd<4, i32>(a.i[0] << count, a.i[1] << count, a.i[2] << count, a.i[3] << count);
#endif
        }
        static inline Simd<4, i32> shr(const Simd<4, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_srli_epi32(a.v128, count));
#else
            return Simd<4, i32>(
                (i32)(((u32)a.i[0]) >> count),
                (i32)(((u32)a.i[1]) >> count),
                (i32)(((u32)a.i[2]) >> count),
                (i32)(((u32)a.i[3]) >> count));
#endif
        }
        static inline Simd<4, i32> sar(const Simd<4, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_srai_epi32(a.v128, count));
#else
            return Simd<4, i32>(a.i[0] >> count, a.i[1] >> count, a.i[2] >> count, a.i[3] >> count);
#endif
        }

        static inline Simd<4, i32> mul(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<4, i32>(_mm_mullo_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(a.i[0] * b.i[0], a.i[1] * b.i[1], a.i[2] * b.i[2], a.i[3] * b.i[3]);
#endif
        }

        static inline Simd<4, i32> cmpEq(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_cmpeq_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, i32> cmpNe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_xor_si128(_mm_cmpeq_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
#else
            return Simd<4, i32>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, i32> cmpLt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_cmplt_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, i32> cmpLe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_xor_si128(_mm_cmpgt_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
#else
            return Simd<4, i32>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, i32> cmpGt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_cmpgt_epi32(a.v128, b.v128));
#else
            return Simd<4, i32>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, i32> cmpGe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, i32>(_mm_xor_si128(_mm_cmplt_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
#else
            return Simd<4, i32>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0);
#endif
        }

        static inline u32 movemask(const Simd<4, i32> &a) {
#if ATOM_SIMD_SSE2
            return (u32)_mm_movemask_epi8(a.v128);
#else
            return (u32)_mm_movemask_ps(a.v128);
#endif
        }
    };

    template <>
    union Simd<8, i32> {
#if ATOM_SIMD_SSE2
        __m128i v128[2];
#else
        __m128 v128[2];
#endif
        i32 i[8];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128i a0, __m128i a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
#else
        Simd(__m128 a0, __m128 a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
#endif
        Simd(i32 a0, i32 a1, i32 a2, i32 a3, i32 a4, i32 a5, i32 a6, i32 a7) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
            i[4] = a4;
            i[5] = a5;
            i[6] = a6;
            i[7] = a7;
        }

        static inline Simd<8, i32> load(const i32* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_loadu_si128((const __m128i*)ptr), _mm_loadu_si128((const __m128i*)(ptr + 4)));
#else
            return Simd<8, i32>(_mm_loadu_ps((const f32*)ptr), _mm_loadu_ps((const f32*)ptr + 4));
#endif
        }
        static inline void store(i32* ptr, const Simd<8, i32> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_si128((__m128i*)ptr, simd.v128[0]);
            _mm_storeu_si128((__m128i*)(ptr + 4), simd.v128[1]);
#else
            _mm_storeu_ps((f32*)ptr, simd.v128[0]);
            _mm_storeu_ps((f32*)(ptr + 4), simd.v128[1]);
#endif
        }
        static inline Simd<8, i32> set1(const i32 value) {
#if ATOM_SIMD_SSE2
            __m128i v = _mm_set1_epi32(value);
            return Simd<8, i32>(v, v);
#else
            __m128 v = _mm_set1_ps((f32)value);
            return Simd<8, i32>(v, v);
#endif
        }
        static inline Simd<8, i32>
            set(const i32 a0,
                const i32 a1,
                const i32 a2,
                const i32 a3,
                const i32 a4,
                const i32 a5,
                const i32 a6,
                const i32 a7) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(_mm_set_epi32(a3, a2, a1, a0), _mm_set_epi32(a7, a6, a5, a4));
#else
            return Simd<8, i32>(
                _mm_set_ps((f32)a3, (f32)a2, (f32)a1, (f32)a0),
                _mm_set_ps((f32)a7, (f32)a6, (f32)a5, (f32)a4));
#endif
        }
        static inline Simd<8, i32> add(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_add_epi32(a.v128[0], b.v128[0]), _mm_add_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] + b.i[0],
                a.i[1] + b.i[1],
                a.i[2] + b.i[2],
                a.i[3] + b.i[3],
                a.i[4] + b.i[4],
                a.i[5] + b.i[5],
                a.i[6] + b.i[6],
                a.i[7] + b.i[7]);
#endif
        }
        static inline Simd<8, i32> sub(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_sub_epi32(a.v128[0], b.v128[0]), _mm_sub_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] - b.i[0],
                a.i[1] - b.i[1],
                a.i[2] - b.i[2],
                a.i[3] - b.i[3],
                a.i[4] - b.i[4],
                a.i[5] - b.i[5],
                a.i[6] - b.i[6],
                a.i[7] - b.i[7]);
#endif
        }
        static inline Simd<8, i32> abs(const Simd<8, i32> &a) {
#if ATOM_SIMD_SSSE3
            return Simd<8, i32>(_mm_abs_epi32(a.v128[0]), _mm_abs_epi32(a.v128[1]));
#elif ATOM_SIMD_SSE2
            __m128i sign0 = _mm_srai_epi32(a.v128[0], 31);
            __m128i sign1 = _mm_srai_epi32(a.v128[1], 31);
            return Simd<8, i32>(
                _mm_sub_epi32(_mm_xor_si128(a.v128[0], sign0), sign0),
                _mm_sub_epi32(_mm_xor_si128(a.v128[1], sign1), sign1));
#else
            return Simd<8, i32>(
                labs(a.i[0]),
                labs(a.i[1]),
                labs(a.i[2]),
                labs(a.i[3]),
                labs(a.i[4]),
                labs(a.i[5]),
                labs(a.i[6]),
                labs(a.i[7]));
#endif
        }
        static inline Simd<8, i32> neg(const Simd<8, i32> &a) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[0]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[1]));
#else
            return Simd<8, i32>(
                -a.i[0], -a.i[1], -a.i[2], -a.i[3], -a.i[4], -a.i[5], -a.i[6], -a.i[7]);
#endif
        }
        static inline Simd<8, i32> min(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<8, i32>(
                _mm_min_epi32(a.v128[0], b.v128[0]), _mm_min_epi32(a.v128[1], b.v128[1]));
#elif ATOM_SIMD_SSE2
            __m128i mask0 = _mm_cmplt_epi32(a.v128[0], b.v128[0]);
            __m128i mask1 = _mm_cmplt_epi32(a.v128[1], b.v128[1]);
            return Simd<8, i32>(
                _mm_or_si128(_mm_and_si128(mask0, a.v128[0]), _mm_andnot_si128(mask0, b.v128[0])),
                _mm_or_si128(_mm_and_si128(mask1, a.v128[1]), _mm_andnot_si128(mask1, b.v128[1])));
#else
            return Simd<8, i32>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3],
                a.i[4] < b.i[4] ? a.i[4] : b.i[4],
                a.i[5] < b.i[5] ? a.i[5] : b.i[5],
                a.i[6] < b.i[6] ? a.i[6] : b.i[6],
                a.i[7] < b.i[7] ? a.i[7] : b.i[7]);
#endif
        }
        static inline Simd<8, i32> max(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<8, i32>(
                _mm_max_epi32(a.v128[0], b.v128[0]), _mm_max_epi32(a.v128[1], b.v128[1]));
#elif ATOM_SIMD_SSE2
            __m128i mask0 = _mm_cmpgt_epi32(a.v128[0], b.v128[0]);
            __m128i mask1 = _mm_cmpgt_epi32(a.v128[1], b.v128[1]);
            return Simd<8, i32>(
                _mm_or_si128(_mm_and_si128(mask0, a.v128[0]), _mm_andnot_si128(mask0, b.v128[0])),
                _mm_or_si128(_mm_and_si128(mask1, a.v128[1]), _mm_andnot_si128(mask1, b.v128[1])));
#else
            return Simd<8, i32>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3],
                a.i[4] > b.i[4] ? a.i[4] : b.i[4],
                a.i[5] > b.i[5] ? a.i[5] : b.i[5],
                a.i[6] > b.i[6] ? a.i[6] : b.i[6],
                a.i[7] > b.i[7] ? a.i[7] : b.i[7]);
#endif
        }
        static inline Simd<8, i32> land(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_and_si128(a.v128[0], b.v128[0]), _mm_and_si128(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(_mm_and_ps(a.v128[0], b.v128[0]), _mm_and_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> lor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_or_si128(a.v128[0], b.v128[0]), _mm_or_si128(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(_mm_or_ps(a.v128[0], b.v128[0]), _mm_or_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> lxor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_xor_si128(a.v128[0], b.v128[0]), _mm_xor_si128(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(_mm_xor_ps(a.v128[0], b.v128[0]), _mm_xor_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> lnot(const Simd<8, i32> &a) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(_mm_xor_si128(a.v128[0], ones), _mm_xor_si128(a.v128[1], ones));
#else
            __m128 ones = _mm_set_ps1(~0);
            return Simd<8, i32>(_mm_xor_ps(a.v128[0], ones), _mm_xor_ps(a.v128[1], ones));
#endif
        }
        static inline Simd<8, i32> landnot(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_andnot_si128(b.v128[0], a.v128[0]), _mm_andnot_si128(b.v128[1], a.v128[1]));
#else
            return Simd<8, i32>(
                _mm_andnot_ps(b.v128[0], a.v128[0]), _mm_andnot_ps(b.v128[1], a.v128[1]));
#endif
        }
        static inline Simd<8, i32> shl(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(_mm_slli_epi32(a.v128[0], count), _mm_slli_epi32(a.v128[1], count));
#else
            return Simd<8, i32>(
                a.i[0] << count,
                a.i[1] << count,
                a.i[2] << count,
                a.i[3] << count,
                a.i[4] << count,
                a.i[5] << count,
                a.i[6] << count,
                a.i[7] << count);
#endif
        }
        static inline Simd<8, i32> shr(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(_mm_srli_epi32(a.v128[0], count), _mm_srli_epi32(a.v128[1], count));
#else
            return Simd<8, i32>(
                (i32)(((u32)a.i[0]) >> count),
                (i32)(((u32)a.i[1]) >> count),
                (i32)(((u32)a.i[2]) >> count),
                (i32)(((u32)a.i[3]) >> count),
                (i32)(((u32)a.i[4]) >> count),
                (i32)(((u32)a.i[5]) >> count),
                (i32)(((u32)a.i[6]) >> count),
                (i32)(((u32)a.i[7]) >> count));
#endif
        }
        static inline Simd<8, i32> sar(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(_mm_srai_epi32(a.v128[0], count), _mm_srai_epi32(a.v128[1], count));
#else
            return Simd<8, i32>(
                a.i[0] >> count,
                a.i[1] >> count,
                a.i[2] >> count,
                a.i[3] >> count,
                a.i[4] >> count,
                a.i[5] >> count,
                a.i[6] >> count,
                a.i[7] >> count);
#endif
        }

        static inline Simd<8, i32> mul(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<8, i32>(
                _mm_mullo_epi32(a.v128[0], b.v128[0]), _mm_mullo_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] * b.i[0],
                a.i[1] * b.i[1],
                a.i[2] * b.i[2],
                a.i[3] * b.i[3],
                a.i[4] * b.i[4],
                a.i[5] * b.i[5],
                a.i[6] * b.i[6],
                a.i[7] * b.i[7]);
#endif
        }

        static inline Simd<8, i32> cmpEq(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_cmpeq_epi32(a.v128[0], b.v128[0]), _mm_cmpeq_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0,
                a.i[4] == b.i[4] ? ~0 : 0,
                a.i[5] == b.i[5] ? ~0 : 0,
                a.i[6] == b.i[6] ? ~0 : 0,
                a.i[7] == b.i[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, i32> cmpNe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[1], b.v128[1]), ones));
#else
            return Simd<8, i32>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0,
                a.i[4] != b.i[4] ? ~0 : 0,
                a.i[5] != b.i[5] ? ~0 : 0,
                a.i[6] != b.i[6] ? ~0 : 0,
                a.i[7] != b.i[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, i32> cmpLt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_cmplt_epi32(a.v128[0], b.v128[0]), _mm_cmplt_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0,
                a.i[4] < b.i[4] ? ~0 : 0,
                a.i[5] < b.i[5] ? ~0 : 0,
                a.i[6] < b.i[6] ? ~0 : 0,
                a.i[7] < b.i[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, i32> cmpLe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[1], b.v128[1]), ones));
#else
            return Simd<8, i32>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0,
                a.i[4] <= b.i[4] ? ~0 : 0,
                a.i[5] <= b.i[5] ? ~0 : 0,
                a.i[6] <= b.i[6] ? ~0 : 0,
                a.i[7] <= b.i[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, i32> cmpGt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, i32>(
                _mm_cmpgt_epi32(a.v128[0], b.v128[0]), _mm_cmpgt_epi32(a.v128[1], b.v128[1]));
#else
            return Simd<8, i32>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0,
                a.i[4] > b.i[4] ? ~0 : 0,
                a.i[5] > b.i[5] ? ~0 : 0,
                a.i[6] > b.i[6] ? ~0 : 0,
                a.i[7] > b.i[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, i32> cmpGe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[1], b.v128[1]), ones));
#else
            return Simd<8, i32>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0,
                a.i[4] >= b.i[4] ? ~0 : 0,
                a.i[5] >= b.i[5] ? ~0 : 0,
                a.i[6] >= b.i[6] ? ~0 : 0,
                a.i[7] >= b.i[7] ? ~0 : 0);
#endif
        }
        static inline u32 movemask(const Simd<8, i32> &a) {
#if ATOM_SIMD_SSE2
            return (u32)_mm_movemask_epi8(a.v128[0]) | ((u32)_mm_movemask_epi8(a.v128[1]) << 16);
#else
            return (u32)_mm_movemask_ps(a.v128[0]) | ((u32)_mm_movemask_ps(a.v128[1]) << 16);
#endif
        }
    };

    template <>
    union Simd<16, i32> {
#if ATOM_SIMD_SSE2
        __m128i v128[4];
#else
        __m128 v128[4];
#endif
        i32 i[16];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128i a0, __m128i a1, __m128i a2, __m128i a3) {
            v128[0] = a0;
            v128[1] = a1;
            v128[2] = a2;
            v128[3] = a3;
        }
#else
        Simd(__m128 a0, __m128 a1, __m128 a2, __m128 a3) {
            v128[0] = a0;
            v128[1] = a1;
            v128[2] = a2;
            v128[3] = a3;
        }
#endif
        Simd(
            i32 a0,
            i32 a1,
            i32 a2,
            i32 a3,
            i32 a4,
            i32 a5,
            i32 a6,
            i32 a7,
            i32 a8,
            i32 a9,
            i32 a10,
            i32 a11,
            i32 a12,
            i32 a13,
            i32 a14,
            i32 a15) {
            i[0]  = a0;
            i[1]  = a1;
            i[2]  = a2;
            i[3]  = a3;
            i[4]  = a4;
            i[5]  = a5;
            i[6]  = a6;
            i[7]  = a7;
            i[8]  = a8;
            i[9]  = a9;
            i[10] = a10;
            i[11] = a11;
            i[12] = a12;
            i[13] = a13;
            i[14] = a14;
            i[15] = a15;
        }

        static inline Simd<16, i32> load(const i32* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_loadu_si128((const __m128i*)ptr),
                _mm_loadu_si128((const __m128i*)(ptr + 4)),
                _mm_loadu_si128((const __m128i*)(ptr + 8)),
                _mm_loadu_si128((const __m128i*)(ptr + 12)));
#else
            return Simd<16, i32>(
                _mm_loadu_ps((const f32*)ptr),
                _mm_loadu_ps((const f32*)ptr + 4),
                _mm_loadu_ps((const f32*)ptr + 8),
                _mm_loadu_ps((const f32*)ptr + 12));
#endif
        }
        static inline void store(i32* ptr, const Simd<16, i32> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_si128((__m128i*)ptr, simd.v128[0]);
            _mm_storeu_si128((__m128i*)(ptr + 4), simd.v128[1]);
            _mm_storeu_si128((__m128i*)(ptr + 8), simd.v128[2]);
            _mm_storeu_si128((__m128i*)(ptr + 12), simd.v128[3]);
#else
            _mm_storeu_ps((f32*)ptr, simd.v128[0]);
            _mm_storeu_ps((f32*)(ptr + 4), simd.v128[1]);
            _mm_storeu_ps((f32*)(ptr + 8), simd.v128[2]);
            _mm_storeu_ps((f32*)(ptr + 12), simd.v128[3]);
#endif
        }
        static inline Simd<16, i32> set1(const i32 value) {
#if ATOM_SIMD_SSE2
            __m128i v = _mm_set1_epi32(value);
            return Simd<16, i32>(v, v, v, v);
#else
            __m128 v = _mm_set1_ps((f32)value);
            return Simd<16, i32>(v, v, v, v);
#endif
        }
        static inline Simd<16, i32>
            set(const i32 a0,
                const i32 a1,
                const i32 a2,
                const i32 a3,
                const i32 a4,
                const i32 a5,
                const i32 a6,
                const i32 a7,
                const i32 a8,
                const i32 a9,
                const i32 a10,
                const i32 a11,
                const i32 a12,
                const i32 a13,
                const i32 a14,
                const i32 a15) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_set_epi32(a3, a2, a1, a0),
                _mm_set_epi32(a7, a6, a5, a4),
                _mm_set_epi32(a11, a10, a9, a8),
                _mm_set_epi32(a15, a14, a13, a12));
#else
            return Simd<16, i32>(
                _mm_set_ps((f32)a3, (f32)a2, (f32)a1, (f32)a0),
                _mm_set_ps((f32)a7, (f32)a6, (f32)a5, (f32)a4),
                _mm_set_ps((f32)a11, (f32)a10, (f32)a9, (f32)a8),
                _mm_set_ps((f32)a15, (f32)a14, (f32)a13, (f32)a12));
#endif
        }

        static inline Simd<16, i32> add(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_add_epi32(a.v128[0], b.v128[0]),
                _mm_add_epi32(a.v128[1], b.v128[1]),
                _mm_add_epi32(a.v128[2], b.v128[2]),
                _mm_add_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] + b.i[0],
                a.i[1] + b.i[1],
                a.i[2] + b.i[2],
                a.i[3] + b.i[3],
                a.i[4] + b.i[4],
                a.i[5] + b.i[5],
                a.i[6] + b.i[6],
                a.i[7] + b.i[7],
                a.i[8] + b.i[8],
                a.i[9] + b.i[9],
                a.i[10] + b.i[10],
                a.i[11] + b.i[11],
                a.i[12] + b.i[12],
                a.i[13] + b.i[13],
                a.i[14] + b.i[14],
                a.i[15] + b.i[15]);
#endif
        }
        static inline Simd<16, i32> sub(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_sub_epi32(a.v128[0], b.v128[0]),
                _mm_sub_epi32(a.v128[1], b.v128[1]),
                _mm_sub_epi32(a.v128[2], b.v128[2]),
                _mm_sub_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] - b.i[0],
                a.i[1] - b.i[1],
                a.i[2] - b.i[2],
                a.i[3] - b.i[3],
                a.i[4] - b.i[4],
                a.i[5] - b.i[5],
                a.i[6] - b.i[6],
                a.i[7] - b.i[7],
                a.i[8] - b.i[8],
                a.i[9] - b.i[9],
                a.i[10] - b.i[10],
                a.i[11] - b.i[11],
                a.i[12] - b.i[12],
                a.i[13] - b.i[13],
                a.i[14] - b.i[14],
                a.i[15] - b.i[15]);
#endif
        }
        static inline Simd<16, i32> abs(const Simd<16, i32> &a) {
#if ATOM_SIMD_SSSE3
            return Simd<16, i32>(
                _mm_abs_epi32(a.v128[0]),
                _mm_abs_epi32(a.v128[1]),
                _mm_abs_epi32(a.v128[2]),
                _mm_abs_epi32(a.v128[3]));
#elif ATOM_SIMD_SSE2
            __m128i sign0 = _mm_srai_epi32(a.v128[0], 31);
            __m128i sign1 = _mm_srai_epi32(a.v128[1], 31);
            __m128i sign2 = _mm_srai_epi32(a.v128[2], 31);
            __m128i sign3 = _mm_srai_epi32(a.v128[3], 31);
            return Simd<16, i32>(
                _mm_sub_epi32(_mm_xor_si128(a.v128[0], sign0), sign0),
                _mm_sub_epi32(_mm_xor_si128(a.v128[1], sign1), sign1),
                _mm_sub_epi32(_mm_xor_si128(a.v128[2], sign2), sign2),
                _mm_sub_epi32(_mm_xor_si128(a.v128[3], sign3), sign3));
#else
            return Simd<16, i32>(
                labs(a.i[0]),
                labs(a.i[1]),
                labs(a.i[2]),
                labs(a.i[3]),
                labs(a.i[4]),
                labs(a.i[5]),
                labs(a.i[6]),
                labs(a.i[7]),
                labs(a.i[8]),
                labs(a.i[9]),
                labs(a.i[10]),
                labs(a.i[11]),
                labs(a.i[12]),
                labs(a.i[13]),
                labs(a.i[14]),
                labs(a.i[15]));
#endif
        }
        static inline Simd<16, i32> neg(const Simd<16, i32> &a) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[0]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[1]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[2]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[3]));
#else
            return Simd<16, i32>(
                -a.i[0],
                -a.i[1],
                -a.i[2],
                -a.i[3],
                -a.i[4],
                -a.i[5],
                -a.i[6],
                -a.i[7],
                -a.i[8],
                -a.i[9],
                -a.i[10],
                -a.i[11],
                -a.i[12],
                -a.i[13],
                -a.i[14],
                -a.i[15]);
#endif
        }
        static inline Simd<16, i32> min(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<16, i32>(
                _mm_min_epi32(a.v128[0], b.v128[0]),
                _mm_min_epi32(a.v128[1], b.v128[1]),
                _mm_min_epi32(a.v128[2], b.v128[2]),
                _mm_min_epi32(a.v128[3], b.v128[3]));
#elif ATOM_SIMD_SSE2
            __m128i mask0 = _mm_cmplt_epi32(a.v128[0], b.v128[0]);
            __m128i mask1 = _mm_cmplt_epi32(a.v128[1], b.v128[1]);
            __m128i mask2 = _mm_cmplt_epi32(a.v128[2], b.v128[2]);
            __m128i mask3 = _mm_cmplt_epi32(a.v128[3], b.v128[3]);
            return Simd<16, i32>(
                _mm_or_si128(_mm_and_si128(mask0, a.v128[0]), _mm_andnot_si128(mask0, b.v128[0])),
                _mm_or_si128(_mm_and_si128(mask1, a.v128[1]), _mm_andnot_si128(mask1, b.v128[1])),
                _mm_or_si128(_mm_and_si128(mask2, a.v128[2]), _mm_andnot_si128(mask2, b.v128[2])),
                _mm_or_si128(_mm_and_si128(mask3, a.v128[3]), _mm_andnot_si128(mask3, b.v128[3])));
#else
            return Simd<16, i32>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3],
                a.i[4] < b.i[4] ? a.i[4] : b.i[4],
                a.i[5] < b.i[5] ? a.i[5] : b.i[5],
                a.i[6] < b.i[6] ? a.i[6] : b.i[6],
                a.i[7] < b.i[7] ? a.i[7] : b.i[7],
                a.i[8] < b.i[8] ? a.i[8] : b.i[8],
                a.i[9] < b.i[9] ? a.i[9] : b.i[9],
                a.i[10] < b.i[10] ? a.i[10] : b.i[10],
                a.i[11] < b.i[11] ? a.i[11] : b.i[11],
                a.i[12] < b.i[12] ? a.i[12] : b.i[12],
                a.i[13] < b.i[13] ? a.i[13] : b.i[13],
                a.i[14] < b.i[14] ? a.i[14] : b.i[14],
                a.i[15] < b.i[15] ? a.i[15] : b.i[15]);
#endif
        }
        static inline Simd<16, i32> max(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<16, i32>(
                _mm_max_epi32(a.v128[0], b.v128[0]),
                _mm_max_epi32(a.v128[1], b.v128[1]),
                _mm_max_epi32(a.v128[2], b.v128[2]),
                _mm_max_epi32(a.v128[3], b.v128[3]));
#elif ATOM_SIMD_SSE2
            __m128i mask0 = _mm_cmpgt_epi32(a.v128[0], b.v128[0]);
            __m128i mask1 = _mm_cmpgt_epi32(a.v128[1], b.v128[1]);
            __m128i mask2 = _mm_cmpgt_epi32(a.v128[2], b.v128[2]);
            __m128i mask3 = _mm_cmpgt_epi32(a.v128[3], b.v128[3]);
            return Simd<16, i32>(
                _mm_or_si128(_mm_and_si128(mask0, a.v128[0]), _mm_andnot_si128(mask0, b.v128[0])),
                _mm_or_si128(_mm_and_si128(mask1, a.v128[1]), _mm_andnot_si128(mask1, b.v128[1])),
                _mm_or_si128(_mm_and_si128(mask2, a.v128[2]), _mm_andnot_si128(mask2, b.v128[2])),
                _mm_or_si128(_mm_and_si128(mask3, a.v128[3]), _mm_andnot_si128(mask3, b.v128[3])));
#else
            return Simd<16, i32>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3],
                a.i[4] > b.i[4] ? a.i[4] : b.i[4],
                a.i[5] > b.i[5] ? a.i[5] : b.i[5],
                a.i[6] > b.i[6] ? a.i[6] : b.i[6],
                a.i[7] > b.i[7] ? a.i[7] : b.i[7],
                a.i[8] > b.i[8] ? a.i[8] : b.i[8],
                a.i[9] > b.i[9] ? a.i[9] : b.i[9],
                a.i[10] > b.i[10] ? a.i[10] : b.i[10],
                a.i[11] > b.i[11] ? a.i[11] : b.i[11],
                a.i[12] > b.i[12] ? a.i[12] : b.i[12],
                a.i[13] > b.i[13] ? a.i[13] : b.i[13],
                a.i[14] > b.i[14] ? a.i[14] : b.i[14],
                a.i[15] > b.i[15] ? a.i[15] : b.i[15]);
#endif
        }

        static inline Simd<16, i32> land(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_and_si128(a.v128[0], b.v128[0]),
                _mm_and_si128(a.v128[1], b.v128[1]),
                _mm_and_si128(a.v128[2], b.v128[2]),
                _mm_and_si128(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                _mm_and_ps(a.v128[0], b.v128[0]),
                _mm_and_ps(a.v128[1], b.v128[1]),
                _mm_and_ps(a.v128[2], b.v128[2]),
                _mm_and_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> lor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_or_si128(a.v128[0], b.v128[0]),
                _mm_or_si128(a.v128[1], b.v128[1]),
                _mm_or_si128(a.v128[2], b.v128[2]),
                _mm_or_si128(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                _mm_or_ps(a.v128[0], b.v128[0]),
                _mm_or_ps(a.v128[1], b.v128[1]),
                _mm_or_ps(a.v128[2], b.v128[2]),
                _mm_or_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> lxor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_xor_si128(a.v128[0], b.v128[0]),
                _mm_xor_si128(a.v128[1], b.v128[1]),
                _mm_xor_si128(a.v128[2], b.v128[2]),
                _mm_xor_si128(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                _mm_xor_ps(a.v128[0], b.v128[0]),
                _mm_xor_ps(a.v128[1], b.v128[1]),
                _mm_xor_ps(a.v128[2], b.v128[2]),
                _mm_xor_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> lnot(const Simd<16, i32> &a) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(a.v128[0], ones),
                _mm_xor_si128(a.v128[1], ones),
                _mm_xor_si128(a.v128[2], ones),
                _mm_xor_si128(a.v128[3], ones));
#else
            __m128 ones = _mm_set_ps1(~0);
            return Simd<16, i32>(
                _mm_xor_ps(a.v128[0], ones),
                _mm_xor_ps(a.v128[1], ones),
                _mm_xor_ps(a.v128[2], ones),
                _mm_xor_ps(a.v128[3], ones));
#endif
        }
        static inline Simd<16, i32> landnot(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_andnot_si128(b.v128[0], a.v128[0]),
                _mm_andnot_si128(b.v128[1], a.v128[1]),
                _mm_andnot_si128(b.v128[2], a.v128[2]),
                _mm_andnot_si128(b.v128[3], a.v128[3]));
#else
            return Simd<16, i32>(
                _mm_andnot_ps(b.v128[0], a.v128[0]),
                _mm_andnot_ps(b.v128[1], a.v128[1]),
                _mm_andnot_ps(b.v128[2], a.v128[2]),
                _mm_andnot_ps(b.v128[3], a.v128[3]));
#endif
        }

        static inline Simd<16, i32> shl(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_slli_epi32(a.v128[0], count),
                _mm_slli_epi32(a.v128[1], count),
                _mm_slli_epi32(a.v128[2], count),
                _mm_slli_epi32(a.v128[3], count));
#else
            return Simd<16, i32>(
                a.i[0] << count,
                a.i[1] << count,
                a.i[2] << count,
                a.i[3] << count,
                a.i[4] << count,
                a.i[5] << count,
                a.i[6] << count,
                a.i[7] << count,
                a.i[8] << count,
                a.i[9] << count,
                a.i[10] << count,
                a.i[11] << count,
                a.i[12] << count,
                a.i[13] << count,
                a.i[14] << count,
                a.i[15] << count);
#endif
        }
        static inline Simd<16, i32> shr(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_srli_epi32(a.v128[0], count),
                _mm_srli_epi32(a.v128[1], count),
                _mm_srli_epi32(a.v128[2], count),
                _mm_srli_epi32(a.v128[3], count));
#else
            return Simd<16, i32>(
                (i32)(((u32)a.i[0]) >> count),
                (i32)(((u32)a.i[1]) >> count),
                (i32)(((u32)a.i[2]) >> count),
                (i32)(((u32)a.i[3]) >> count),
                (i32)(((u32)a.i[4]) >> count),
                (i32)(((u32)a.i[5]) >> count),
                (i32)(((u32)a.i[6]) >> count),
                (i32)(((u32)a.i[7]) >> count),
                (i32)(((u32)a.i[8]) >> count),
                (i32)(((u32)a.i[9]) >> count),
                (i32)(((u32)a.i[10]) >> count),
                (i32)(((u32)a.i[11]) >> count),
                (i32)(((u32)a.i[12]) >> count),
                (i32)(((u32)a.i[13]) >> count),
                (i32)(((u32)a.i[14]) >> count),
                (i32)(((u32)a.i[15]) >> count));
#endif
        }
        static inline Simd<16, i32> sar(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_srai_epi32(a.v128[0], count),
                _mm_srai_epi32(a.v128[1], count),
                _mm_srai_epi32(a.v128[2], count),
                _mm_srai_epi32(a.v128[3], count));
#else
            return Simd<16, i32>(
                a.i[0] >> count,
                a.i[1] >> count,
                a.i[2] >> count,
                a.i[3] >> count,
                a.i[4] >> count,
                a.i[5] >> count,
                a.i[6] >> count,
                a.i[7] >> count,
                a.i[8] >> count,
                a.i[9] >> count,
                a.i[10] >> count,
                a.i[11] >> count,
                a.i[12] >> count,
                a.i[13] >> count,
                a.i[14] >> count,
                a.i[15] >> count);
#endif
        }

        static inline Simd<16, i32> mul(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE41
            return Simd<16, i32>(
                _mm_mullo_epi32(a.v128[0], b.v128[0]),
                _mm_mullo_epi32(a.v128[1], b.v128[1]),
                _mm_mullo_epi32(a.v128[2], b.v128[2]),
                _mm_mullo_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] * b.i[0],
                a.i[1] * b.i[1],
                a.i[2] * b.i[2],
                a.i[3] * b.i[3],
                a.i[4] * b.i[4],
                a.i[5] * b.i[5],
                a.i[6] * b.i[6],
                a.i[7] * b.i[7],
                a.i[8] * b.i[8],
                a.i[9] * b.i[9],
                a.i[10] * b.i[10],
                a.i[11] * b.i[11],
                a.i[12] * b.i[12],
                a.i[13] * b.i[13],
                a.i[14] * b.i[14],
                a.i[15] * b.i[15]);
#endif
        }

        static inline Simd<16, i32> cmpEq(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_cmpeq_epi32(a.v128[0], b.v128[0]),
                _mm_cmpeq_epi32(a.v128[1], b.v128[1]),
                _mm_cmpeq_epi32(a.v128[2], b.v128[2]),
                _mm_cmpeq_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0,
                a.i[4] == b.i[4] ? ~0 : 0,
                a.i[5] == b.i[5] ? ~0 : 0,
                a.i[6] == b.i[6] ? ~0 : 0,
                a.i[7] == b.i[7] ? ~0 : 0,
                a.i[8] == b.i[8] ? ~0 : 0,
                a.i[9] == b.i[9] ? ~0 : 0,
                a.i[10] == b.i[10] ? ~0 : 0,
                a.i[11] == b.i[11] ? ~0 : 0,
                a.i[12] == b.i[12] ? ~0 : 0,
                a.i[13] == b.i[13] ? ~0 : 0,
                a.i[14] == b.i[14] ? ~0 : 0,
                a.i[15] == b.i[15] ? ~0 : 0);
#endif
        }
        static inline Simd<16, i32> cmpNe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[3], b.v128[3]), ones));
#else
            return Simd<16, i32>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0,
                a.i[4] != b.i[4] ? ~0 : 0,
                a.i[5] != b.i[5] ? ~0 : 0,
                a.i[6] != b.i[6] ? ~0 : 0,
                a.i[7] != b.i[7] ? ~0 : 0,
                a.i[8] != b.i[8] ? ~0 : 0,
                a.i[9] != b.i[9] ? ~0 : 0,
                a.i[10] != b.i[10] ? ~0 : 0,
                a.i[11] != b.i[11] ? ~0 : 0,
                a.i[12] != b.i[12] ? ~0 : 0,
                a.i[13] != b.i[13] ? ~0 : 0,
                a.i[14] != b.i[14] ? ~0 : 0,
                a.i[15] != b.i[15] ? ~0 : 0);
#endif
        }
        static inline Simd<16, i32> cmpLt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_cmplt_epi32(a.v128[0], b.v128[0]),
                _mm_cmplt_epi32(a.v128[1], b.v128[1]),
                _mm_cmplt_epi32(a.v128[2], b.v128[2]),
                _mm_cmplt_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0,
                a.i[4] < b.i[4] ? ~0 : 0,
                a.i[5] < b.i[5] ? ~0 : 0,
                a.i[6] < b.i[6] ? ~0 : 0,
                a.i[7] < b.i[7] ? ~0 : 0,
                a.i[8] < b.i[8] ? ~0 : 0,
                a.i[9] < b.i[9] ? ~0 : 0,
                a.i[10] < b.i[10] ? ~0 : 0,
                a.i[11] < b.i[11] ? ~0 : 0,
                a.i[12] < b.i[12] ? ~0 : 0,
                a.i[13] < b.i[13] ? ~0 : 0,
                a.i[14] < b.i[14] ? ~0 : 0,
                a.i[15] < b.i[15] ? ~0 : 0);
#endif
        }
        static inline Simd<16, i32> cmpLe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[3], b.v128[3]), ones));
#else
            return Simd<16, i32>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0,
                a.i[4] <= b.i[4] ? ~0 : 0,
                a.i[5] <= b.i[5] ? ~0 : 0,
                a.i[6] <= b.i[6] ? ~0 : 0,
                a.i[7] <= b.i[7] ? ~0 : 0,
                a.i[8] <= b.i[8] ? ~0 : 0,
                a.i[9] <= b.i[9] ? ~0 : 0,
                a.i[10] <= b.i[10] ? ~0 : 0,
                a.i[11] <= b.i[11] ? ~0 : 0,
                a.i[12] <= b.i[12] ? ~0 : 0,
                a.i[13] <= b.i[13] ? ~0 : 0,
                a.i[14] <= b.i[14] ? ~0 : 0,
                a.i[15] <= b.i[15] ? ~0 : 0);
#endif
        }
        static inline Simd<16, i32> cmpGt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            return Simd<16, i32>(
                _mm_cmpgt_epi32(a.v128[0], b.v128[0]),
                _mm_cmpgt_epi32(a.v128[1], b.v128[1]),
                _mm_cmpgt_epi32(a.v128[2], b.v128[2]),
                _mm_cmpgt_epi32(a.v128[3], b.v128[3]));
#else
            return Simd<16, i32>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0,
                a.i[4] > b.i[4] ? ~0 : 0,
                a.i[5] > b.i[5] ? ~0 : 0,
                a.i[6] > b.i[6] ? ~0 : 0,
                a.i[7] > b.i[7] ? ~0 : 0,
                a.i[8] > b.i[8] ? ~0 : 0,
                a.i[9] > b.i[9] ? ~0 : 0,
                a.i[10] > b.i[10] ? ~0 : 0,
                a.i[11] > b.i[11] ? ~0 : 0,
                a.i[12] > b.i[12] ? ~0 : 0,
                a.i[13] > b.i[13] ? ~0 : 0,
                a.i[14] > b.i[14] ? ~0 : 0,
                a.i[15] > b.i[15] ? ~0 : 0);
#endif
        }
        static inline Simd<16, i32> cmpGe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_SSE2
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[3], b.v128[3]), ones));
#else
            return Simd<16, i32>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0,
                a.i[4] >= b.i[4] ? ~0 : 0,
                a.i[5] >= b.i[5] ? ~0 : 0,
                a.i[6] >= b.i[6] ? ~0 : 0,
                a.i[7] >= b.i[7] ? ~0 : 0,
                a.i[8] >= b.i[8] ? ~0 : 0,
                a.i[9] >= b.i[9] ? ~0 : 0,
                a.i[10] >= b.i[10] ? ~0 : 0,
                a.i[11] >= b.i[11] ? ~0 : 0,
                a.i[12] >= b.i[12] ? ~0 : 0,
                a.i[13] >= b.i[13] ? ~0 : 0,
                a.i[14] >= b.i[14] ? ~0 : 0,
                a.i[15] >= b.i[15] ? ~0 : 0);
#endif
        }

        static inline u64 movemask(const Simd<16, i32> &a) {
#if ATOM_SIMD_SSE2
            return (u64)_mm_movemask_epi8(a.v128[0])
                 | ((u64)_mm_movemask_epi8(a.v128[1]) << 16)
                 | ((u64)_mm_movemask_epi8(a.v128[2]) << 32)
                 | ((u64)_mm_movemask_epi8(a.v128[3]) << 48);
#else
            return (u64)_mm_movemask_ps(a.v128[0])
                 | ((u64)_mm_movemask_ps(a.v128[1]) << 16)
                 | ((u64)_mm_movemask_ps(a.v128[2]) << 32)
                 | ((u64)_mm_movemask_ps(a.v128[3]) << 48);
#endif
        }
    };

}

#endif
