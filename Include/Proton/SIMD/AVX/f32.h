#ifndef AXN_PRTN_SIMD_AVX_F32_H
#define AXN_PRTN_SIMD_AVX_F32_H

/**
 * @file SIMD/AVX/f32.h
 * @brief AVX/SSE implementations for 32-bit float SIMD (`Proton::Simd<4,f32>`).
 *
 * Uses AVX/SSE intrinsics to implement 128-bit float vector operations and
 * extensions when appropriate.
 */

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include "../Simd.h"
#include <Quark/Standard.h>

namespace Proton {
    template <>
    union Simd<4, f32> {
        __m128 v128;
        f32    v32[4];
        Simd() {}
        Simd(__m128 a0) { v128 = a0; }
        Simd(f32 a0, f32 a1, f32 a2, f32 a3) {
            v32[0] = a0;
            v32[1] = a1;
            v32[2] = a2;
            v32[3] = a3;
        }

        static inline Simd<4, f32> ATOM_NOEXCEPT(load(const f32* ptr)) {
            return Simd<4, f32>(_mm_loadu_ps(ptr));
        }
        static inline void ATOM_NOEXCEPT(store(f32* ptr, const Simd<4, f32> &simd)) {
            _mm_storeu_ps(ptr, simd.v128);
        }
        static inline Simd<4, f32> set1(const f32 value) {
            return Simd<4, f32>(_mm_set1_ps(value));
        }
        static inline Simd<4, f32> set(const f32 a0, const f32 a1, const f32 a2, const f32 a3) {
            return Simd<4, f32>(_mm_set_ps(a3, a2, a1, a0));
        }

        static inline Simd<4, f32> add(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_add_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> sub(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_sub_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> abs(const Simd<4, f32> &a) {
            __m128 mask = _mm_set_ps1(-0.0f);
            return Simd<4, f32>(_mm_andnot_ps(mask, a.v128));
        }
        static inline Simd<4, f32> neg(const Simd<4, f32> &a) {
            return Simd<4, f32>(_mm_xor_ps(a.v128, _mm_set1_ps(-0.0f)));
        }
        static inline Simd<4, f32> min(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_min_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> max(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_max_ps(a.v128, b.v128));
        }

        static inline Simd<4, f32> land(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_and_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> lor(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_or_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> lxor(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_xor_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> lnot(const Simd<4, f32> &a) {
            __m128 ones = _mm_castsi128_ps(_mm_set1_epi32(~0));
            return Simd<4, f32>(_mm_xor_ps(a.v128, ones));
        }
        static inline Simd<4, f32> landnot(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_andnot_ps(b.v128, a.v128));
        }

        static inline Simd<4, f32> mul(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_mul_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> div(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_div_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> sqrt(const Simd<4, f32> &a) {
            return Simd<4, f32>(_mm_sqrt_ps(a.v128));
        }
        static inline Simd<4, f32> rsqrt(const Simd<4, f32> &a) {
            return Simd<4, f32>(_mm_rsqrt_ps(a.v128));
        }
        static inline Simd<4, f32> reciprocal(const Simd<4, f32> &a) {
            return Simd<4, f32>(_mm_rcp_ps(a.v128));
        }

        static inline Simd<4, f32> cmpEq(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmpeq_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> cmpNe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmpneq_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> cmpLt(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmplt_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> cmpLe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmple_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> cmpGt(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmpgt_ps(a.v128, b.v128));
        }
        static inline Simd<4, f32> cmpGe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(_mm_cmpge_ps(a.v128, b.v128));
        }

        static inline u32 movemask(const Simd<4, f32> &a) { return (u32)_mm_movemask_ps(a.v128); }
    };

    template <>
    union Simd<8, f32> {
        __m256 v256;
        __m128 v128[2];
        f32    v32[8];
        Simd() {}
        Simd(__m256 a0) { v256 = a0; }
        Simd(__m128 a0, __m128 a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
        Simd(f32 a0, f32 a1, f32 a2, f32 a3, f32 a4, f32 a5, f32 a6, f32 a7) {
            v32[0] = a0;
            v32[1] = a1;
            v32[2] = a2;
            v32[3] = a3;
            v32[4] = a4;
            v32[5] = a5;
            v32[6] = a6;
            v32[7] = a7;
        }

        static inline Simd<8, f32> load(const f32* ptr) {
            return Simd<8, f32>(_mm256_loadu_ps(ptr));
        }
        static inline void store(f32* ptr, const Simd<8, f32> &simd) {
            _mm256_storeu_ps(ptr, simd.v256);
        }
        static inline Simd<8, f32> set1(const f32 value) {
            return Simd<8, f32>(_mm256_set1_ps(value));
        }
        static inline Simd<8, f32>
            set(const f32 a0,
                const f32 a1,
                const f32 a2,
                const f32 a3,
                const f32 a4,
                const f32 a5,
                const f32 a6,
                const f32 a7) {
            return Simd<8, f32>(_mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0));
        }

        static inline Simd<8, f32> add(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_add_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> sub(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_sub_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> abs(const Simd<8, f32> &a) {
            __m256 mask = _mm256_set1_ps(-0.0f);
            return Simd<8, f32>(_mm256_andnot_ps(mask, a.v256));
        }
        static inline Simd<8, f32> neg(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm256_xor_ps(a.v256, _mm256_set1_ps(-0.0f)));
        }
        static inline Simd<8, f32> min(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_min_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> max(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_max_ps(a.v256, b.v256));
        }

        static inline Simd<8, f32> land(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_and_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> lor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_or_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> lxor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_xor_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> lnot(const Simd<8, f32> &a) {
            __m256 ones = _mm256_castsi256_ps(_mm256_set1_epi32(~0));
            return Simd<8, f32>(_mm256_xor_ps(a.v256, ones));
        }
        static inline Simd<8, f32> landnot(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_andnot_ps(b.v256, a.v256));
        }

        static inline Simd<8, f32> mul(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_mul_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> div(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_div_ps(a.v256, b.v256));
        }
        static inline Simd<8, f32> sqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm256_sqrt_ps(a.v256));
        }
        static inline Simd<8, f32> rsqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm256_rsqrt_ps(a.v256));
        }
        static inline Simd<8, f32> reciprocal(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm256_rcp_ps(a.v256));
        }

        static inline Simd<8, f32> cmpEq(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_EQ_OQ));
        }
        static inline Simd<8, f32> cmpNe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_NEQ_OQ));
        }
        static inline Simd<8, f32> cmpLt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_LT_OQ));
        }
        static inline Simd<8, f32> cmpLe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_LE_OQ));
        }
        static inline Simd<8, f32> cmpGt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_GT_OQ));
        }
        static inline Simd<8, f32> cmpGe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm256_cmp_ps(a.v256, b.v256, _CMP_GE_OQ));
        }

        static inline u32 movemask(const Simd<8, f32> &a) {
            return (u32)_mm256_movemask_ps(a.v256);
        }
    };
    template <>
    union Simd<16, f32> {
#if ATOM_SIMD_AVX512F
        __m512 v512;
#endif
        __m256 v256[2];
        __m128 v128[4];
        f32    v32[16];
        Simd() {}
#if ATOM_SIMD_AVX512F
        Simd(__m512 a0) { v512 = a0; }
#endif
        Simd(__m256 a0, __m256 a1) {
            v256[0] = a0;
            v256[1] = a1;
        }
        Simd(__m128 a0, __m128 a1, __m128 a2, __m128 a3) {
            v128[0] = a0;
            v128[1] = a1;
            v128[2] = a2;
            v128[3] = a3;
        }
        Simd(
            f32 a0,
            f32 a1,
            f32 a2,
            f32 a3,
            f32 a4,
            f32 a5,
            f32 a6,
            f32 a7,
            f32 a8,
            f32 a9,
            f32 a10,
            f32 a11,
            f32 a12,
            f32 a13,
            f32 a14,
            f32 a15) {
            v32[0]  = a0;
            v32[1]  = a1;
            v32[2]  = a2;
            v32[3]  = a3;
            v32[4]  = a4;
            v32[5]  = a5;
            v32[6]  = a6;
            v32[7]  = a7;
            v32[8]  = a8;
            v32[9]  = a9;
            v32[10] = a10;
            v32[11] = a11;
            v32[12] = a12;
            v32[13] = a13;
            v32[14] = a14;
            v32[15] = a15;
        }

        static inline Simd<16, f32> load(const f32* ptr) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_loadu_ps(ptr));
#else
            return Simd<16, f32>(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr + 8));
#endif
        }
        static inline void store(f32* ptr, const Simd<16, f32> &simd) {
#if ATOM_SIMD_AVX512F
            _mm512_storeu_ps(ptr, simd.v512);
#else
            _mm256_storeu_ps(ptr, simd.v256[0]);
            _mm256_storeu_ps(ptr + 8, simd.v256[1]);
#endif
        }
        static inline Simd<16, f32> set1(const f32 value) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_set1_ps(value));
#else
            __m256 v = _mm256_set1_ps(value);
            return Simd<16, f32>(v, v);
#endif
        }
        static inline Simd<16, f32>
            set(const f32 a0,
                const f32 a1,
                const f32 a2,
                const f32 a3,
                const f32 a4,
                const f32 a5,
                const f32 a6,
                const f32 a7,
                const f32 a8,
                const f32 a9,
                const f32 a10,
                const f32 a11,
                const f32 a12,
                const f32 a13,
                const f32 a14,
                const f32 a15) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_set_ps(
                a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0));
#else
            return Simd<16, f32>(
                _mm256_set_ps(a7, a6, a5, a4, a3, a2, a1, a0),
                _mm256_set_ps(a15, a14, a13, a12, a11, a10, a9, a8));
#endif
        }

        static inline Simd<16, f32> add(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_add_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_add_ps(a.v256[0], b.v256[0]), _mm256_add_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> sub(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_sub_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_sub_ps(a.v256[0], b.v256[0]), _mm256_sub_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> abs(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_abs_ps(a.v512));
#else
            __m256 mask = _mm256_set1_ps(-0.0f);
            return Simd<16, f32>(
                _mm256_andnot_ps(mask, a.v256[0]), _mm256_andnot_ps(mask, a.v256[1]));
#endif
        }
        static inline Simd<16, f32> neg(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_xor_ps(a.v512, _mm512_set1_ps(-0.0f)));
#else
            __m256 sign = _mm256_set1_ps(-0.0f);
            return Simd<16, f32>(_mm256_xor_ps(a.v256[0], sign), _mm256_xor_ps(a.v256[1], sign));
#endif
        }
        static inline Simd<16, f32> min(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_min_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_min_ps(a.v256[0], b.v256[0]), _mm256_min_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> max(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_max_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_max_ps(a.v256[0], b.v256[0]), _mm256_max_ps(a.v256[1], b.v256[1]));
#endif
        }

        static inline Simd<16, f32> land(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_and_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_and_ps(a.v256[0], b.v256[0]), _mm256_and_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> lor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_or_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_or_ps(a.v256[0], b.v256[0]), _mm256_or_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> lxor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_xor_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_xor_ps(a.v256[0], b.v256[0]), _mm256_xor_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> lnot(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_castsi512_ps(_mm512_ternarylogic_epi32(
                _mm512_castps_si512(a.v512),
                _mm512_castps_si512(a.v512),
                _mm512_castps_si512(a.v512),
                0x0F)));
#else
            __m256 ones = _mm256_castsi256_ps(_mm256_set1_epi32(~0));
            return Simd<16, f32>(_mm256_xor_ps(a.v256[0], ones), _mm256_xor_ps(a.v256[1], ones));
#endif
        }
        static inline Simd<16, f32> landnot(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_castsi512_ps(_mm512_ternarylogic_epi32(
                _mm512_castps_si512(a.v512),
                _mm512_castps_si512(b.v512),
                _mm512_castps_si512(b.v512),
                0x40)));
#else
            return Simd<16, f32>(
                _mm256_andnot_ps(b.v256[0], a.v256[0]), _mm256_andnot_ps(b.v256[1], a.v256[1]));
#endif
        }

        static inline Simd<16, f32> mul(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_mul_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_mul_ps(a.v256[0], b.v256[0]), _mm256_mul_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> div(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_div_ps(a.v512, b.v512));
#else
            return Simd<16, f32>(
                _mm256_div_ps(a.v256[0], b.v256[0]), _mm256_div_ps(a.v256[1], b.v256[1]));
#endif
        }
        static inline Simd<16, f32> sqrt(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_sqrt_ps(a.v512));
#else
            return Simd<16, f32>(_mm256_sqrt_ps(a.v256[0]), _mm256_sqrt_ps(a.v256[1]));
#endif
        }
        static inline Simd<16, f32> rsqrt(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_rsqrt14_ps(a.v512));
#else
            return Simd<16, f32>(_mm256_rsqrt_ps(a.v256[0]), _mm256_rsqrt_ps(a.v256[1]));
#endif
        }
        static inline Simd<16, f32> reciprocal(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512F
            return Simd<16, f32>(_mm512_rcp14_ps(a.v512));
#else
            return Simd<16, f32>(_mm256_rcp_ps(a.v256[0]), _mm256_rcp_ps(a.v256[1]));
#endif
        }

        static inline Simd<16, f32> cmpEq(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_EQ_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_EQ_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_EQ_OQ));
#endif
        }
        static inline Simd<16, f32> cmpNe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_NEQ_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_NEQ_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_NEQ_OQ));
#endif
        }
        static inline Simd<16, f32> cmpLt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_LT_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_LT_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_LT_OQ));
#endif
        }
        static inline Simd<16, f32> cmpLe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_LE_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_LE_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_LE_OQ));
#endif
        }
        static inline Simd<16, f32> cmpGt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_GT_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_GT_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_GT_OQ));
#endif
        }
        static inline Simd<16, f32> cmpGe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
#if ATOM_SIMD_AVX512F
            __mmask16 k = _mm512_cmp_ps_mask(a.v512, b.v512, _CMP_GE_OQ);
            return Simd<16, f32>(_mm512_maskz_mov_ps(k, _mm512_set1_ps(-1.0f)));
#else
            return Simd<16, f32>(
                _mm256_cmp_ps(a.v256[0], b.v256[0], _CMP_GE_OQ),
                _mm256_cmp_ps(a.v256[1], b.v256[1], _CMP_GE_OQ));
#endif
        }

        static inline u64 movemask(const Simd<16, f32> &a) {
#if ATOM_SIMD_AVX512BW
            return (u64)_mm512_movepi8_mask(a.v512);
#else
            return (u64)_mm256_movemask_ps(a.v256[0]) | ((u64)_mm256_movemask_ps(a.v256[1]) << 8);
#endif
        }
    };
}

#endif
