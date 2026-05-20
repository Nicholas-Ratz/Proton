#ifndef ATOM_PRTN_SIMD_SSE_F32_H
#define ATOM_PRTN_SIMD_SSE_F32_H

/**
 * @file SIMD/SSE/f32.h
 * @brief SSE implementation for 32-bit float SIMD (`Proton::Simd<4,f32>`).
 *
 * Uses SSE intrinsics to implement load/store, arithmetic, bitwise and
 * comparison operations for 128-bit float vectors.
 */

#include <xmmintrin.h>

#include "../Simd.h"
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<4, f32> {
        __m128 v128;
        f32    f[4];
        Simd() {}
        Simd(__m128 a0) { v128 = a0; }
        Simd(f32 a0, f32 a1, f32 a2, f32 a3) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
        }

        static inline Simd<4, f32> load(const f32* ptr) { return Simd<4, f32>(_mm_loadu_ps(ptr)); }
        static inline void         store(f32* ptr, const Simd<4, f32> &simd) {
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
            return Simd<4, f32>(_mm_xor_ps(a.v128, _mm_set_ps1(-0.0f)));
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
            return Simd<4, f32>(_mm_xor_ps(a.v128, _mm_set_ps1(~0)));
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
        __m128 v128[2];
        f32    f[8];
        Simd() {}
        Simd(__m128 a0, __m128 a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
        Simd(f32 a0, f32 a1, f32 a2, f32 a3, f32 a4, f32 a5, f32 a6, f32 a7) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
            f[4] = a4;
            f[5] = a5;
            f[6] = a6;
            f[7] = a7;
        }

        static inline Simd<8, f32> load(const f32* ptr) {
            return Simd<8, f32>(_mm_loadu_ps(ptr), _mm_loadu_ps(ptr + 4));
        }
        static inline void store(f32* ptr, const Simd<8, f32> &simd) {
            _mm_storeu_ps(ptr, simd.v128[0]);
            _mm_storeu_ps(ptr + 4, simd.v128[1]);
        }
        static inline Simd<8, f32> set1(const f32 value) {
            __m128 v = _mm_set1_ps(value);
            return Simd<8, f32>(v, v);
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
            return Simd<8, f32>(_mm_set_ps(a3, a2, a1, a0), _mm_set_ps(a7, a6, a5, a4));
        }

        static inline Simd<8, f32> add(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_add_ps(a.v128[0], b.v128[0]), _mm_add_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> sub(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_sub_ps(a.v128[0], b.v128[0]), _mm_sub_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> abs(const Simd<8, f32> &a) {
            __m128 mask = _mm_set_ps1(-0.0f);
            return Simd<8, f32>(_mm_andnot_ps(mask, a.v128[0]), _mm_andnot_ps(mask, a.v128[1]));
        }
        static inline Simd<8, f32> neg(const Simd<8, f32> &a) {
            __m128 mask = _mm_set_ps1(-0.0f);
            return Simd<8, f32>(_mm_xor_ps(a.v128[0], mask), _mm_xor_ps(a.v128[1], mask));
        }
        static inline Simd<8, f32> min(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_min_ps(a.v128[0], b.v128[0]), _mm_min_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> max(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_max_ps(a.v128[0], b.v128[0]), _mm_max_ps(a.v128[1], b.v128[1]));
        }

        static inline Simd<8, f32> land(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_and_ps(a.v128[0], b.v128[0]), _mm_and_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> lor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_or_ps(a.v128[0], b.v128[0]), _mm_or_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> lxor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_xor_ps(a.v128[0], b.v128[0]), _mm_xor_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> lnot(const Simd<8, f32> &a) {
            __m128 ones = _mm_set_ps1(~0);
            return Simd<8, f32>(_mm_xor_ps(a.v128[0], ones), _mm_xor_ps(a.v128[1], ones));
        }
        static inline Simd<8, f32> landnot(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_andnot_ps(b.v128[0], a.v128[0]), _mm_andnot_ps(b.v128[1], a.v128[1]));
        }

        static inline Simd<8, f32> mul(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_mul_ps(a.v128[0], b.v128[0]), _mm_mul_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> div(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(_mm_div_ps(a.v128[0], b.v128[0]), _mm_div_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> sqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm_sqrt_ps(a.v128[0]), _mm_sqrt_ps(a.v128[1]));
        }
        static inline Simd<8, f32> rsqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm_rsqrt_ps(a.v128[0]), _mm_rsqrt_ps(a.v128[1]));
        }
        static inline Simd<8, f32> reciprocal(const Simd<8, f32> &a) {
            return Simd<8, f32>(_mm_rcp_ps(a.v128[0]), _mm_rcp_ps(a.v128[1]));
        }

        static inline Simd<8, f32> cmpEq(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmpeq_ps(a.v128[0], b.v128[0]), _mm_cmpeq_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> cmpNe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmpneq_ps(a.v128[0], b.v128[0]), _mm_cmpneq_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> cmpLt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmplt_ps(a.v128[0], b.v128[0]), _mm_cmplt_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> cmpLe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmple_ps(a.v128[0], b.v128[0]), _mm_cmple_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> cmpGt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmpgt_ps(a.v128[0], b.v128[0]), _mm_cmpgt_ps(a.v128[1], b.v128[1]));
        }
        static inline Simd<8, f32> cmpGe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                _mm_cmpge_ps(a.v128[0], b.v128[0]), _mm_cmpge_ps(a.v128[1], b.v128[1]));
        }

        static inline u32 movemask(const Simd<8, f32> &a) {
            return (u32)_mm_movemask_ps(a.v128[0]) | (u32)(_mm_movemask_ps(a.v128[1]) << 16);
        }
    };

    template <>
    union Simd<16, f32> {
        __m128 v128[4];
        f32    f[16];
        Simd() {}
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
            f[0]  = a0;
            f[1]  = a1;
            f[2]  = a2;
            f[3]  = a3;
            f[4]  = a4;
            f[5]  = a5;
            f[6]  = a6;
            f[7]  = a7;
            f[8]  = a8;
            f[9]  = a9;
            f[10] = a10;
            f[11] = a11;
            f[12] = a12;
            f[13] = a13;
            f[14] = a14;
            f[15] = a15;
        }

        static inline Simd<16, f32> load(const f32* ptr) {
            return Simd<16, f32>(
                _mm_loadu_ps(ptr),
                _mm_loadu_ps(ptr + 4),
                _mm_loadu_ps(ptr + 8),
                _mm_loadu_ps(ptr + 12));
        }
        static inline void store(f32* ptr, const Simd<16, f32> &simd) {
            _mm_storeu_ps(ptr, simd.v128[0]);
            _mm_storeu_ps(ptr + 4, simd.v128[1]);
            _mm_storeu_ps(ptr + 8, simd.v128[2]);
            _mm_storeu_ps(ptr + 12, simd.v128[3]);
        }
        static inline Simd<16, f32> set1(const f32 value) {
            __m128 v = _mm_set1_ps(value);
            return Simd<16, f32>(v, v, v, v);
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
            return Simd<16, f32>(
                _mm_set_ps(a3, a2, a1, a0),
                _mm_set_ps(a7, a6, a5, a4),
                _mm_set_ps(a11, a10, a9, a8),
                _mm_set_ps(a15, a14, a13, a12));
        }

        static inline Simd<16, f32> add(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_add_ps(a.v128[0], b.v128[0]),
                _mm_add_ps(a.v128[1], b.v128[1]),
                _mm_add_ps(a.v128[2], b.v128[2]),
                _mm_add_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> sub(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_sub_ps(a.v128[0], b.v128[0]),
                _mm_sub_ps(a.v128[1], b.v128[1]),
                _mm_sub_ps(a.v128[2], b.v128[2]),
                _mm_sub_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> abs(const Simd<16, f32> &a) {
            __m128 mask = _mm_set_ps1(-0.0f);
            return Simd<16, f32>(
                _mm_andnot_ps(mask, a.v128[0]),
                _mm_andnot_ps(mask, a.v128[1]),
                _mm_andnot_ps(mask, a.v128[2]),
                _mm_andnot_ps(mask, a.v128[3]));
        }
        static inline Simd<16, f32> neg(const Simd<16, f32> &a) {
            __m128 mask = _mm_set_ps1(-0.0f);
            return Simd<16, f32>(
                _mm_xor_ps(a.v128[0], mask),
                _mm_xor_ps(a.v128[1], mask),
                _mm_xor_ps(a.v128[2], mask),
                _mm_xor_ps(a.v128[3], mask));
        }
        static inline Simd<16, f32> min(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_min_ps(a.v128[0], b.v128[0]),
                _mm_min_ps(a.v128[1], b.v128[1]),
                _mm_min_ps(a.v128[2], b.v128[2]),
                _mm_min_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> max(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_max_ps(a.v128[0], b.v128[0]),
                _mm_max_ps(a.v128[1], b.v128[1]),
                _mm_max_ps(a.v128[2], b.v128[2]),
                _mm_max_ps(a.v128[3], b.v128[3]));
        }

        static inline Simd<16, f32> land(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_and_ps(a.v128[0], b.v128[0]),
                _mm_and_ps(a.v128[1], b.v128[1]),
                _mm_and_ps(a.v128[2], b.v128[2]),
                _mm_and_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> lor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_or_ps(a.v128[0], b.v128[0]),
                _mm_or_ps(a.v128[1], b.v128[1]),
                _mm_or_ps(a.v128[2], b.v128[2]),
                _mm_or_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> lxor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_xor_ps(a.v128[0], b.v128[0]),
                _mm_xor_ps(a.v128[1], b.v128[1]),
                _mm_xor_ps(a.v128[2], b.v128[2]),
                _mm_xor_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> lnot(const Simd<16, f32> &a) {
            __m128 ones = _mm_set_ps1(~0);
            return Simd<16, f32>(
                _mm_xor_ps(a.v128[0], ones),
                _mm_xor_ps(a.v128[1], ones),
                _mm_xor_ps(a.v128[2], ones),
                _mm_xor_ps(a.v128[3], ones));
        }
        static inline Simd<16, f32> landnot(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_andnot_ps(b.v128[0], a.v128[0]),
                _mm_andnot_ps(b.v128[1], a.v128[1]),
                _mm_andnot_ps(b.v128[2], a.v128[2]),
                _mm_andnot_ps(b.v128[3], a.v128[3]));
        }

        static inline Simd<16, f32> mul(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_mul_ps(a.v128[0], b.v128[0]),
                _mm_mul_ps(a.v128[1], b.v128[1]),
                _mm_mul_ps(a.v128[2], b.v128[2]),
                _mm_mul_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> div(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_div_ps(a.v128[0], b.v128[0]),
                _mm_div_ps(a.v128[1], b.v128[1]),
                _mm_div_ps(a.v128[2], b.v128[2]),
                _mm_div_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> sqrt(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                _mm_sqrt_ps(a.v128[0]),
                _mm_sqrt_ps(a.v128[1]),
                _mm_sqrt_ps(a.v128[2]),
                _mm_sqrt_ps(a.v128[3]));
        }
        static inline Simd<16, f32> rsqrt(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                _mm_rsqrt_ps(a.v128[0]),
                _mm_rsqrt_ps(a.v128[1]),
                _mm_rsqrt_ps(a.v128[2]),
                _mm_rsqrt_ps(a.v128[3]));
        }
        static inline Simd<16, f32> reciprocal(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                _mm_rcp_ps(a.v128[0]),
                _mm_rcp_ps(a.v128[1]),
                _mm_rcp_ps(a.v128[2]),
                _mm_rcp_ps(a.v128[3]));
        }

        static inline Simd<16, f32> cmpEq(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmpeq_ps(a.v128[0], b.v128[0]),
                _mm_cmpeq_ps(a.v128[1], b.v128[1]),
                _mm_cmpeq_ps(a.v128[2], b.v128[2]),
                _mm_cmpeq_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> cmpNe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmpneq_ps(a.v128[0], b.v128[0]),
                _mm_cmpneq_ps(a.v128[1], b.v128[1]),
                _mm_cmpneq_ps(a.v128[2], b.v128[2]),
                _mm_cmpneq_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> cmpLt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmplt_ps(a.v128[0], b.v128[0]),
                _mm_cmplt_ps(a.v128[1], b.v128[1]),
                _mm_cmplt_ps(a.v128[2], b.v128[2]),
                _mm_cmplt_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> cmpLe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmple_ps(a.v128[0], b.v128[0]),
                _mm_cmple_ps(a.v128[1], b.v128[1]),
                _mm_cmple_ps(a.v128[2], b.v128[2]),
                _mm_cmple_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> cmpGt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmpgt_ps(a.v128[0], b.v128[0]),
                _mm_cmpgt_ps(a.v128[1], b.v128[1]),
                _mm_cmpgt_ps(a.v128[2], b.v128[2]),
                _mm_cmpgt_ps(a.v128[3], b.v128[3]));
        }
        static inline Simd<16, f32> cmpGe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                _mm_cmpge_ps(a.v128[0], b.v128[0]),
                _mm_cmpge_ps(a.v128[1], b.v128[1]),
                _mm_cmpge_ps(a.v128[2], b.v128[2]),
                _mm_cmpge_ps(a.v128[3], b.v128[3]));
        }

        static inline u64 movemask(const Simd<16, f32> &a) {
            return (u64)_mm_movemask_ps(a.v128[0])
                 | ((u64)_mm_movemask_ps(a.v128[1]) << 16)
                 | ((u64)_mm_movemask_ps(a.v128[2]) << 32)
                 | ((u64)_mm_movemask_ps(a.v128[3]) << 48);
        }
    };

}
#endif
