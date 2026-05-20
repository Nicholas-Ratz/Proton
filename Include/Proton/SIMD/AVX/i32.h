#ifndef AXN_PRTN_SIMD_AVX_I32_H
#define AXN_PRTN_SIMD_AVX_I32_H

/**
 * @file SIMD/AVX/i32.h
 * @brief AVX/SSE implementations for 32-bit integer SIMD (`Proton::Simd<4,i32>`).
 *
 * Uses AVX intrinsics for integer arithmetic, bitwise ops and comparisons.
 */

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include "../Simd.h"
#include <Quark/System/Simd.h> // IWYU pragma: keep
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<4, i32> {
        __m128i v128;
        i32     i[4];
        Simd() {}
        Simd(__m128i a0) { v128 = a0; }
        Simd(i32 a0, i32 a1, i32 a2, i32 a3) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
        }

        static inline Simd<4, i32> load(const i32* ptr) {
            return Simd<4, i32>(_mm_loadu_si128((const __m128i_u*)ptr));
        }
        static inline void store(i32* ptr, const Simd<4, i32> &simd) {
            _mm_storeu_si128((__m128i_u*)ptr, simd.v128);
        }
        static inline Simd<4, i32> set1(const i32 value) {
            return Simd<4, i32>(_mm_set1_epi32(value));
        }
        static inline Simd<4, i32> set(const i32 a0, const i32 a1, const i32 a2, const i32 a3) {
            return Simd<4, i32>(_mm_set_epi32(a3, a2, a1, a0));
        }

        static inline Simd<4, i32> add(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_add_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> sub(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_sub_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> abs(const Simd<4, i32> &a) {
            return Simd<4, i32>(_mm_abs_epi32(a.v128));
        }
        static inline Simd<4, i32> neg(const Simd<4, i32> &a) {
            return Simd<4, i32>(_mm_sub_epi32(_mm_setzero_si128(), a.v128));
        }
        static inline Simd<4, i32> min(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_min_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> max(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_max_epi32(a.v128, b.v128));
        }

        static inline Simd<4, i32> land(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_and_si128(a.v128, b.v128));
        }
        static inline Simd<4, i32> lor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_or_si128(a.v128, b.v128));
        }
        static inline Simd<4, i32> lxor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_xor_si128(a.v128, b.v128));
        }
        static inline Simd<4, i32> lnot(const Simd<4, i32> &a) {
            return Simd<4, i32>(_mm_xor_si128(a.v128, _mm_set1_epi32(~0)));
        }
        static inline Simd<4, i32> landnot(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_andnot_si128(b.v128, a.v128));
        }

        static inline Simd<4, i32> shl(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(_mm_slli_epi32(a.v128, count));
        }
        static inline Simd<4, i32> shr(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(_mm_srli_epi32(a.v128, count));
        }
        static inline Simd<4, i32> sar(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(_mm_srai_epi32(a.v128, count));
        }

        static inline Simd<4, i32> mul(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_mullo_epi32(a.v128, b.v128));
        }

        static inline Simd<4, i32> cmpEq(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_cmpeq_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> cmpNe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_xor_si128(_mm_cmpeq_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
        }
        static inline Simd<4, i32> cmpLt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_cmplt_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> cmpLe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_xor_si128(_mm_cmpgt_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
        }
        static inline Simd<4, i32> cmpGt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_cmpgt_epi32(a.v128, b.v128));
        }
        static inline Simd<4, i32> cmpGe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(_mm_xor_si128(_mm_cmplt_epi32(a.v128, b.v128), _mm_set1_epi32(~0)));
        }

        static inline u32 movemask(const Simd<4, i32> &a) { return (u32)_mm_movemask_epi8(a.v128); }
    };

    template <>
    union Simd<8, i32> {
        __m256i v256;
        __m128i v128[2];
        i32     i[8];
        Simd() {}
        Simd(__m256i a0) { v256 = a0; }
        Simd(__m128i a0, __m128i a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
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
            return Simd<8, i32>(_mm256_loadu_si256((const __m256i_u*)ptr));
        }
        static inline void store(i32* ptr, const Simd<8, i32> &simd) {
            _mm256_storeu_si256((__m256i_u*)ptr, simd.v256);
        }
        static inline Simd<8, i32> set1(const i32 value) {
            return Simd<8, i32>(_mm256_set1_epi32(value));
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
            return Simd<8, i32>(_mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0));
        }

        static inline Simd<8, i32> add(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_add_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_add_epi32(a.v128[0], b.v128[0]), _mm_add_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> sub(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_sub_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_sub_epi32(a.v128[0], b.v128[0]), _mm_sub_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> abs(const Simd<8, i32> &a) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_abs_epi32(a.v256));
#else
            return Simd<8, i32>(_mm_abs_epi32(a.v128[0]), _mm_abs_epi32(a.v128[1]));
#endif
        }
        static inline Simd<8, i32> neg(const Simd<8, i32> &a) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_sub_epi32(_mm256_setzero_si256(), a.v256));
#else
            return Simd<8, i32>(
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[0]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[1]));
#endif
        }
        static inline Simd<8, i32> min(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_min_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_min_epi32(a.v128[0], b.v128[0]), _mm_min_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> max(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_max_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_max_epi32(a.v128[0], b.v128[0]), _mm_max_epi32(a.v128[1], b.v128[1]));
#endif
        }

        static inline Simd<8, i32> land(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_and_si256(a.v256, b.v256));
#else
            return Simd<8, i32>(_mm256_castps_si256(
                _mm256_and_ps(_mm256_castsi256_ps(a.v256), _mm256_castsi256_ps(b.v256))));
#endif
        }
        static inline Simd<8, i32> lor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_or_si256(a.v256, b.v256));
#else
            return Simd<8, i32>(_mm256_castps_si256(
                _mm256_or_ps(_mm256_castsi256_ps(a.v256), _mm256_castsi256_ps(b.v256))));
#endif
        }
        static inline Simd<8, i32> lxor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_xor_si256(a.v256, b.v256));
#else
            return Simd<8, i32>(_mm256_castps_si256(
                _mm256_xor_ps(_mm256_castsi256_ps(a.v256), _mm256_castsi256_ps(b.v256))));
#endif
        }
        static inline Simd<8, i32> lnot(const Simd<8, i32> &a) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_xor_si256(a.v256, _mm256_set1_epi32(~0)));
#else
            return Simd<8, i32>(_mm256_castps_si256(_mm256_xor_ps(
                _mm256_castsi256_ps(a.v256), _mm256_castsi256_ps(_mm256_set1_epi32(~0)))));
#endif
        }
        static inline Simd<8, i32> landnot(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_andnot_si256(b.v256, a.v256));
#else
            return Simd<8, i32>(_mm256_castps_si256(
                _mm256_andnot_ps(_mm256_castsi256_ps(b.v256), _mm256_castsi256_ps(a.v256))));
#endif
        }

        static inline Simd<8, i32> shl(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_slli_epi32(a.v256, count));
#else
            return Simd<8, i32>(_mm_slli_epi32(a.v128[0], count), _mm_slli_epi32(a.v128[1], count));
#endif
        }
        static inline Simd<8, i32> shr(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_srli_epi32(a.v256, count));
#else
            return Simd<8, i32>(_mm_srli_epi32(a.v128[0], count), _mm_srli_epi32(a.v128[1], count));
#endif
        }
        static inline Simd<8, i32> sar(const Simd<8, i32> &a, i32 count) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_srai_epi32(a.v256, count));
#else
            return Simd<8, i32>(_mm_srai_epi32(a.v128[0], count), _mm_srai_epi32(a.v128[1], count));
#endif
        }

        static inline Simd<8, i32> mul(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_mullo_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_mullo_epi32(a.v128[0], b.v128[0]), _mm_mullo_epi32(a.v128[1], b.v128[1]));
#endif
        }

        static inline Simd<8, i32> cmpEq(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_cmpeq_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_cmpeq_epi32(a.v128[0], b.v128[0]), _mm_cmpeq_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> cmpNe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(
                _mm256_xor_si256(_mm256_cmpeq_epi32(a.v256, b.v256), _mm256_set1_epi32(~0)));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[1], b.v128[1]), ones));
#endif
        }
        static inline Simd<8, i32> cmpLt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_cmpgt_epi32(b.v256, a.v256));
#else
            return Simd<8, i32>(
                _mm_cmplt_epi32(a.v128[0], b.v128[0]), _mm_cmplt_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> cmpLe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(
                _mm256_xor_si256(_mm256_cmpgt_epi32(a.v256, b.v256), _mm256_set1_epi32(~0)));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[1], b.v128[1]), ones));
#endif
        }
        static inline Simd<8, i32> cmpGt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(_mm256_cmpgt_epi32(a.v256, b.v256));
#else
            return Simd<8, i32>(
                _mm_cmpgt_epi32(a.v128[0], b.v128[0]), _mm_cmpgt_epi32(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<8, i32> cmpGe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
#if ATOM_SIMD_AVX2
            return Simd<8, i32>(
                _mm256_xor_si256(_mm256_cmpgt_epi32(b.v256, a.v256), _mm256_set1_epi32(~0)));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<8, i32>(
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[1], b.v128[1]), ones));
#endif
        }

        static inline u32 movemask(const Simd<8, i32> &a) {
#if ATOM_SIMD_AVX2
            return (u32)_mm256_movemask_epi8(a.v256);
#else
            return (u32)_mm256_movemask_ps(_mm256_castsi256_ps(a.v256));
#endif
        }
    };

    template <>
    union Simd<16, i32> {
#if ATOM_SIMD_AVX512F
        __m512i v512;
#endif
        __m256i v256[2];
        __m128i v128[4];
        i32     i[16];
        Simd() {}
#if ATOM_SIMD_AVX512F
        Simd(__m512i a0) { v512 = a0; }
#endif
        Simd(__m256i a0, __m256i a1) {
            v256[0] = a0;
            v256[1] = a1;
        }
        Simd(__m128i a0, __m128i a1, __m128i a2, __m128i a3) {
            v128[0] = a0;
            v128[1] = a1;
            v128[2] = a2;
            v128[3] = a3;
        }
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
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_loadu_si512((const void*)ptr));
#else
            return Simd<16, i32>(
                _mm256_loadu_si256((const __m256i_u*)ptr),
                _mm256_loadu_si256((const __m256i_u*)(ptr + 8)));
#endif
        }
        static inline void store(i32* ptr, const Simd<16, i32> &simd) {
#if ATOM_SIMD_AVX512F
            _mm512_storeu_si512((void*)ptr, simd.v512);
#else
            _mm256_storeu_si256((__m256i_u*)ptr, simd.v256[0]);
            _mm256_storeu_si256((__m256i_u*)(ptr + 8), simd.v256[1]);
#endif
        }
        static inline Simd<16, i32> set1(const i32 value) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_set1_epi32(value));
#else
            __m256i v = _mm256_set1_epi32(value);
            return Simd<16, i32>(v, v);
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
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_set_epi32(
                a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0));
#else
            return Simd<16, i32>(
                _mm256_set_epi32(a7, a6, a5, a4, a3, a2, a1, a0),
                _mm256_set_epi32(a15, a14, a13, a12, a11, a10, a9, a8));
#endif
        }
        static inline Simd<16, i32> add(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_add_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_add_epi32(a.v256[0], b.v256[0]), _mm256_add_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_add_epi32(a.v128[0], b.v128[0]),
                _mm_add_epi32(a.v128[1], b.v128[1]),
                _mm_add_epi32(a.v128[2], b.v128[2]),
                _mm_add_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> sub(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_sub_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_sub_epi32(a.v256[0], b.v256[0]), _mm256_sub_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_sub_epi32(a.v128[0], b.v128[0]),
                _mm_sub_epi32(a.v128[1], b.v128[1]),
                _mm_sub_epi32(a.v128[2], b.v128[2]),
                _mm_sub_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> abs(const Simd<16, i32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_abs_epi32(a.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(_mm256_abs_epi32(a.v256[0]), _mm256_abs_epi32(a.v256[1]));
#else
            return Simd<16, i32>(
                _mm_abs_epi32(a.v128[0]),
                _mm_abs_epi32(a.v128[1]),
                _mm_abs_epi32(a.v128[2]),
                _mm_abs_epi32(a.v128[3]));
#endif
        }
        static inline Simd<16, i32> neg(const Simd<16, i32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_sub_epi32(_mm512_setzero_si512(), a.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_sub_epi32(_mm256_setzero_si256(), a.v256[0]),
                _mm256_sub_epi32(_mm256_setzero_si256(), a.v256[1]));
#else
            return Simd<16, i32>(
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[0]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[1]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[2]),
                _mm_sub_epi32(_mm_setzero_si128(), a.v128[3]));
#endif
        }
        static inline Simd<16, i32> min(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_min_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_min_epi32(a.v256[0], b.v256[0]), _mm256_min_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_min_epi32(a.v128[0], b.v128[0]),
                _mm_min_epi32(a.v128[1], b.v128[1]),
                _mm_min_epi32(a.v128[2], b.v128[2]),
                _mm_min_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> max(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_max_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_max_epi32(a.v256[0], b.v256[0]), _mm256_max_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_max_epi32(a.v128[0], b.v128[0]),
                _mm_max_epi32(a.v128[1], b.v128[1]),
                _mm_max_epi32(a.v128[2], b.v128[2]),
                _mm_max_epi32(a.v128[3], b.v128[3]));
#endif
        }

        static inline Simd<16, i32> land(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_and_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_and_si256(a.v256[0], b.v256[0]), _mm256_and_si256(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm256_castps_si256(
                    _mm256_and_ps(_mm256_castsi256_ps(a.v256[0]), _mm256_castsi256_ps(b.v256[0]))),
                _mm256_castps_si256(
                    _mm256_and_ps(_mm256_castsi256_ps(a.v256[1]), _mm256_castsi256_ps(b.v256[1]))));
#endif
        }
        static inline Simd<16, i32> lor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_or_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_or_si256(a.v256[0], b.v256[0]), _mm256_or_si256(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm256_castps_si256(
                    _mm256_or_ps(_mm256_castsi256_ps(a.v256[0]), _mm256_castsi256_ps(b.v256[0]))),
                _mm256_castps_si256(
                    _mm256_or_ps(_mm256_castsi256_ps(a.v256[1]), _mm256_castsi256_ps(b.v256[1]))));
#endif
        }
        static inline Simd<16, i32> lxor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_xor_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_xor_si256(a.v256[0], b.v256[0]), _mm256_xor_si256(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm256_castps_si256(
                    _mm256_xor_ps(_mm256_castsi256_ps(a.v256[0]), _mm256_castsi256_ps(b.v256[0]))),
                _mm256_castps_si256(
                    _mm256_xor_ps(_mm256_castsi256_ps(a.v256[1]), _mm256_castsi256_ps(b.v256[1]))));
#endif
        }
        static inline Simd<16, i32> lnot(const Simd<16, i32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_ternarylogic_epi32(a.v512, a.v512, a.v512, 0x0F));
#elif ATOM_SIMD_AVX2
            __m256i ones = _mm256_set1_epi32(~0);
            return Simd<16, i32>(
                _mm256_xor_si256(a.v256[0], ones), _mm256_xor_si256(a.v256[1], ones));
#else
            __m256i ones = _mm256_set1_epi32(~0);
            return Simd<16, i32>(
                _mm256_castps_si256(
                    _mm256_xor_ps(_mm256_castsi256_ps(a.v256[0]), _mm256_castsi256_ps(ones))),
                _mm256_castps_si256(
                    _mm256_xor_ps(_mm256_castsi256_ps(a.v256[1]), _mm256_castsi256_ps(ones))));
#endif
        }
        static inline Simd<16, i32> landnot(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_ternarylogic_epi32(a.v512, b.v512, b.v512, 0x40));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_andnot_si256(b.v256[0], a.v256[0]),
                _mm256_andnot_si256(b.v256[1], a.v256[1]));
#else
            return Simd<16, i32>(
                _mm256_castps_si256(_mm256_andnot_ps(
                    _mm256_castsi256_ps(b.v256[0]), _mm256_castsi256_ps(a.v256[0]))),
                _mm256_castps_si256(_mm256_andnot_ps(
                    _mm256_castsi256_ps(b.v256[1]), _mm256_castsi256_ps(a.v256[1]))));
#endif
        }

        static inline Simd<16, i32> shl(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_slli_epi32(a.v512, count));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_slli_epi32(a.v256[0], count), _mm256_slli_epi32(a.v256[1], count));
#else
            return Simd<16, i32>(
                _mm_slli_epi32(a.v128[0], count),
                _mm_slli_epi32(a.v128[1], count),
                _mm_slli_epi32(a.v128[2], count),
                _mm_slli_epi32(a.v128[3], count));
#endif
        }
        static inline Simd<16, i32> shr(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_srli_epi32(a.v512, count));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_srli_epi32(a.v256[0], count), _mm256_srli_epi32(a.v256[1], count));
#else
            return Simd<16, i32>(
                _mm_srli_epi32(a.v128[0], count),
                _mm_srli_epi32(a.v128[1], count),
                _mm_srli_epi32(a.v128[2], count),
                _mm_srli_epi32(a.v128[3], count));
#endif
        }
        static inline Simd<16, i32> sar(const Simd<16, i32> &a, i32 count) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_srai_epi32(a.v512, count));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_srai_epi32(a.v256[0], count), _mm256_srai_epi32(a.v256[1], count));
#else
            return Simd<16, i32>(
                _mm_srai_epi32(a.v128[0], count),
                _mm_srai_epi32(a.v128[1], count),
                _mm_srai_epi32(a.v128[2], count),
                _mm_srai_epi32(a.v128[3], count));
#endif
        }

        static inline Simd<16, i32> mul(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, i32>(_mm512_mullo_epi32(a.v512, b.v512));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_mullo_epi32(a.v256[0], b.v256[0]), _mm256_mullo_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_mullo_epi32(a.v128[0], b.v128[0]),
                _mm_mullo_epi32(a.v128[1], b.v128[1]),
                _mm_mullo_epi32(a.v128[2], b.v128[2]),
                _mm_mullo_epi32(a.v128[3], b.v128[3]));
#endif
        }

        static inline Simd<16, i32> cmpEq(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmpeq_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_cmpeq_epi32(a.v256[0], b.v256[0]), _mm256_cmpeq_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_cmpeq_epi32(a.v128[0], b.v128[0]),
                _mm_cmpeq_epi32(a.v128[1], b.v128[1]),
                _mm_cmpeq_epi32(a.v128[2], b.v128[2]),
                _mm_cmpeq_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> cmpNe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmpneq_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            __m256i ones = _mm256_set1_epi32(~0);
            return Simd<16, i32>(
                _mm256_xor_si256(_mm256_cmpeq_epi32(a.v256[0], b.v256[0]), ones),
                _mm256_xor_si256(_mm256_cmpeq_epi32(a.v256[1], b.v256[1]), ones));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmpeq_epi32(a.v128[3], b.v128[3]), ones));
#endif
        }
        static inline Simd<16, i32> cmpLt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmplt_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_cmpgt_epi32(b.v256[0], a.v256[0]), _mm256_cmpgt_epi32(b.v256[1], a.v256[1]));
#else
            return Simd<16, i32>(
                _mm_cmplt_epi32(a.v128[0], b.v128[0]),
                _mm_cmplt_epi32(a.v128[1], b.v128[1]),
                _mm_cmplt_epi32(a.v128[2], b.v128[2]),
                _mm_cmplt_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> cmpLe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmple_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            __m256i ones = _mm256_set1_epi32(~0);
            return Simd<16, i32>(
                _mm256_xor_si256(_mm256_cmpgt_epi32(a.v256[0], b.v256[0]), ones),
                _mm256_xor_si256(_mm256_cmpgt_epi32(a.v256[1], b.v256[1]), ones));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmpgt_epi32(a.v128[3], b.v128[3]), ones));
#endif
        }
        static inline Simd<16, i32> cmpGt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmpgt_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            return Simd<16, i32>(
                _mm256_cmpgt_epi32(a.v256[0], b.v256[0]), _mm256_cmpgt_epi32(a.v256[1], b.v256[1]));
#else
            return Simd<16, i32>(
                _mm_cmpgt_epi32(a.v128[0], b.v128[0]),
                _mm_cmpgt_epi32(a.v128[1], b.v128[1]),
                _mm_cmpgt_epi32(a.v128[2], b.v128[2]),
                _mm_cmpgt_epi32(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<16, i32> cmpGe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmpge_epi32_mask(a.v512, b.v512);
            return Simd<16, i32>(_mm512_maskz_mov_epi32(k, _mm512_set1_epi32(~0)));
#elif ATOM_SIMD_AVX2
            __m256i ones = _mm256_set1_epi32(~0);
            return Simd<16, i32>(
                _mm256_xor_si256(_mm256_cmpgt_epi32(b.v256[0], a.v256[0]), ones),
                _mm256_xor_si256(_mm256_cmpgt_epi32(b.v256[1], a.v256[1]), ones));
#else
            __m128i ones = _mm_set1_epi32(~0);
            return Simd<16, i32>(
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[0], b.v128[0]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[1], b.v128[1]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[2], b.v128[2]), ones),
                _mm_xor_si128(_mm_cmplt_epi32(a.v128[3], b.v128[3]), ones));
#endif
        }

        static inline u64 movemask(const Simd<16, i32> &a) {
#if ATOM_SIMD_AVX512BW
            return (u64)_mm512_movepi8_mask(a.v512);
#elif ATOM_SIMD_AVX2
            return (u64)_mm256_movemask_epi8(a.v256[0])
                 | ((u64)_mm256_movemask_epi8(a.v256[1]) << 32);
#else
            return (u64)_mm256_movemask_ps(_mm256_castsi256_ps(a.v256[0]))
                 | ((u64)_mm256_movemask_ps(_mm256_castsi256_ps(a.v256[1])) << 32);
#endif
        }
    };
}

#endif
