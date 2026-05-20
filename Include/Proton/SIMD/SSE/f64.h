#ifndef ATOM_PRTN_SIMD_SSE_F64_H
#define ATOM_PRTN_SIMD_SSE_F64_H

/**
 * @file SIMD/SSE/f64.h
 * @brief SSE implementation for 64-bit float SIMD (`Proton::Simd<2,f64>`).
 *
 * Implements double-precision vector operations using SSE2 intrinsics.
 */

#include <math.h>

#include "../Simd.h"
#include <Quark/System/Simd.h> // IWYU pragma: keep
#include <Quark/Types.h>

#undef ATOM_SIMD_SSE2

#if ATOM_SIMD_SSE2
#    include <emmintrin.h>
#endif
#include <xmmintrin.h>

namespace Proton {
    template <>
    union Simd<2, f64> {
#if ATOM_SIMD_SSE2
        __m128d v128;
#else
        __m128 v128;
#endif
        f64 f[2];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128d a0) { v128 = a0; }
#else
        Simd(__m128 a0) { v128 = a0; }
#endif
        Simd(f64 a0, f64 a1) {
            f[0] = a0;
            f[1] = a1;
        }

        static inline Simd<2, f64> load(const f64* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_loadu_pd(ptr));
#else
            return Simd<2, f64>(_mm_loadu_ps((f32*)ptr));
#endif
        }
        static inline void store(f64* ptr, const Simd<2, f64> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_pd(ptr, simd.v128);
#else
            _mm_storeu_ps((f32*)ptr, simd.v128);
#endif
        }
        static inline Simd<2, f64> set1(const f64 value) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_set1_pd(value));
#else
            return Simd<2, f64>(_mm_set_ps(
                (f32)((u64)value >> 32), (f32)value, (f32)((u64)value >> 32), (f32)value));
#endif
        }
        static inline Simd<2, f64> set(const f64 a0, const f64 a1) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_set_pd(a1, a0));
#else
            return Simd<2, f64>(
                _mm_set_ps((f32)((u64)a1 >> 32), (f32)a1, (f32)((u64)a0 >> 32), (f32)a0));
#endif
        }

        static inline Simd<2, f64> add(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_add_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] + b.f[0], a.f[1] + b.f[1]);
#endif
        }
        static inline Simd<2, f64> sub(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_sub_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] - b.f[0], a.f[1] - b.f[1]);
#endif
        }
        static inline Simd<2, f64> abs(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d mask = _mm_set1_pd(-0.0);
            return Simd<2, f64>(_mm_andnot_pd(mask, a.v128));
#else
            return Simd<2, f64>(fabs(a.f[0]), fabs(a.f[1]));
#endif
        }
        static inline Simd<2, f64> neg(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_xor_pd(a.v128, _mm_set1_pd(-0.0)));
#else
            return Simd<2, f64>(-a.f[0], -a.f[1]);
#endif
        }
        static inline Simd<2, f64> min(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_min_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0], a.f[1] < b.f[1] ? a.f[1] : b.f[1]);
#endif
        }
        static inline Simd<2, f64> max(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_max_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0], a.f[1] > b.f[1] ? a.f[1] : b.f[1]);
#endif
        }

        static inline Simd<2, f64> land(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_and_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(_mm_and_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<2, f64> lor(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_or_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(_mm_or_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<2, f64> lxor(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_xor_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(_mm_xor_ps(a.v128, b.v128));
#endif
        }
        static inline Simd<2, f64> lnot(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_xor_pd(a.v128, _mm_castsi128_pd(_mm_set1_epi64x(~0))));
#else
            return Simd<2, f64>(_mm_xor_ps(a.v128, _mm_castsi128_ps(_mm_set1_epi32(~0))));
#endif
        }
        static inline Simd<2, f64> landnot(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_andnot_pd(b.v128, a.v128));
#else
            return Simd<2, f64>(_mm_andnot_ps(b.v128, a.v128));
#endif
        }

        static inline Simd<2, f64> mul(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_mul_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] * b.f[0], a.f[1] * b.f[1]);
#endif
        }
        static inline Simd<2, f64> div(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_div_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] / b.f[0], a.f[1] / b.f[1]);
#endif
        }
        static inline Simd<2, f64> sqrt(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_sqrt_pd(a.v128));
#else
            return Simd<2, f64>(::sqrt(a.f[0]), ::sqrt(a.f[1]));
#endif
        }
        static inline Simd<2, f64> rsqrt(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<2, f64>(_mm_div_pd(one, _mm_sqrt_pd(a.v128)));
#else
            return Simd<2, f64>(1.0 / ::sqrt(a.f[0]), 1.0 / ::sqrt(a.f[1]));
#endif
        }
        static inline Simd<2, f64> reciprocal(const Simd<2, f64> &a) {

#include <math.h>
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<2, f64>(_mm_div_pd(one, a.v128));
#else
            return Simd<2, f64>(1.0 / a.f[0], 1.0 / a.f[1]);
#endif
        }

        static inline Simd<2, f64> cmpEq(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmpeq_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] == b.f[0] ? ~0 : 0, a.f[1] == b.f[1] ? ~0 : 0);
#endif
        }
        static inline Simd<2, f64> cmpNe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmpneq_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] != b.f[0] ? ~0 : 0, a.f[1] != b.f[1] ? ~0 : 0);
#endif
        }
        static inline Simd<2, f64> cmpLt(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmplt_pd(b.v128, a.v128));
#else
            return Simd<2, f64>(a.f[0] < b.f[0] ? ~0 : 0, a.f[1] < b.f[1] ? ~0 : 0);
#endif
        }
        static inline Simd<2, f64> cmpLe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmple_pd(b.v128, a.v128));
#else
            return Simd<2, f64>(a.f[0] <= b.f[0] ? ~0 : 0, a.f[1] <= b.f[1] ? ~0 : 0);
#endif
        }
        static inline Simd<2, f64> cmpGt(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmpgt_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] > b.f[0] ? ~0 : 0, a.f[1] > b.f[1] ? ~0 : 0);
#endif
        }
        static inline Simd<2, f64> cmpGe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<2, f64>(_mm_cmpge_pd(a.v128, b.v128));
#else
            return Simd<2, f64>(a.f[0] >= b.f[0] ? ~0 : 0, a.f[1] >= b.f[1] ? ~0 : 0);
#endif
        }

        static inline u32 movemask(const Simd<2, f64> &a) {
#if ATOM_SIMD_SSE2
            return (u32)_mm_movemask_pd(a.v128);
#else
            return (u32)_mm_movemask_ps(a.v128);
#endif
        }
    };

    template <>
    union Simd<4, f64> {
#if ATOM_SIMD_SSE2
        __m128d v128[2];
#else
        __m128 v128[2];
#endif
        f64 f[4];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128d a0, __m128d a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
#else
        Simd(__m128 a0, __m128 a1) {
            v128[0] = a0;
            v128[1] = a1;
        }
#endif
        Simd(f64 a0, f64 a1, f64 a2, f64 a3) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
        }

        static inline Simd<4, f64> load(const f64* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_loadu_pd(ptr), _mm_loadu_pd(ptr + 2));
#else
            return Simd<4, f64>(_mm_loadu_ps((f32*)(ptr)), _mm_loadu_ps((f32*)(ptr + 2)));
#endif
        }
        static inline void store(f64* ptr, const Simd<4, f64> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_pd(ptr, simd.v128[0]);
            _mm_storeu_pd(ptr + 2, simd.v128[1]);
#else
            _mm_storeu_ps((f32*)(ptr), simd.v128[0]);
            _mm_storeu_ps((f32*)(ptr + 2), simd.v128[1]);
#endif
        }
        static inline Simd<4, f64> set1(const f64 value) {
#if ATOM_SIMD_SSE2
            __m128d v = _mm_set1_pd(value);
            return Simd<4, f64>(v, v);
#else
            __m128 v = _mm_set_ps(
                (f32)((u64)value >> 32), (f32)value, (f32)((u64)value >> 32), (f32)value);
            return Simd<4, f64>(v, v);
#endif
        }
        static inline Simd<4, f64> set(const f64 a0, const f64 a1, const f64 a2, const f64 a3) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_set_pd(a1, a0), _mm_set_pd(a3, a2));
#else
            return Simd<4, f64>(
                _mm_set_ps((f32)((u64)a1 >> 32), (f32)a1, (f32)((u64)a0 >> 32), (f32)a0),
                _mm_set_ps((f32)((u64)a3 >> 32), (f32)a3, (f32)((u64)a2 >> 32), (f32)a2));
#endif
        }

        static inline Simd<4, f64> add(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_add_pd(a.v128[0], b.v128[0]), _mm_add_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(a.f[0] + b.f[0], a.f[1] + b.f[1], a.f[2] + b.f[2], a.f[3] + b.f[3]);
#endif
        }
        static inline Simd<4, f64> sub(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_sub_pd(a.v128[0], b.v128[0]), _mm_sub_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(a.f[0] - b.f[0], a.f[1] - b.f[1], a.f[2] - b.f[2], a.f[3] - b.f[3]);
#endif
        }
        static inline Simd<4, f64> abs(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d mask = _mm_set1_pd(-0.0);
            return Simd<4, f64>(_mm_andnot_pd(mask, a.v128[0]), _mm_andnot_pd(mask, a.v128[1]));
#else
            return Simd<4, f64>(fabs(a.f[0]), fabs(a.f[1]), fabs(a.f[2]), fabs(a.f[3]));
#endif
        }
        static inline Simd<4, f64> neg(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d sign_bit = _mm_set1_pd(-0.0);
            return Simd<4, f64>(_mm_xor_pd(a.v128[0], sign_bit), _mm_xor_pd(a.v128[1], sign_bit));
#else
            return Simd<4, f64>(-a.f[0], -a.f[1], -a.f[2], -a.f[3]);
#endif
        }
        static inline Simd<4, f64> min(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_min_pd(a.v128[0], b.v128[0]), _mm_min_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3]);
#endif
        }
        static inline Simd<4, f64> max(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_max_pd(a.v128[0], b.v128[0]), _mm_max_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3]);
#endif
        }

        static inline Simd<4, f64> land(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_and_pd(a.v128[0], b.v128[0]), _mm_and_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(_mm_and_ps(a.v128[0], b.v128[0]), _mm_and_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<4, f64> lor(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_or_pd(a.v128[0], b.v128[0]), _mm_or_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(_mm_or_ps(a.v128[0], b.v128[0]), _mm_or_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<4, f64> lxor(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_xor_pd(a.v128[0], b.v128[0]), _mm_xor_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(_mm_xor_ps(a.v128[0], b.v128[0]), _mm_xor_ps(a.v128[1], b.v128[1]));
#endif
        }
        static inline Simd<4, f64> lnot(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d ones = _mm_castsi128_pd(_mm_set1_epi64x(~0));
            return Simd<4, f64>(_mm_xor_pd(a.v128[0], ones), _mm_xor_pd(a.v128[1], ones));
#else
            __m128 ones = _mm_castsi128_ps(_mm_set1_epi32(~0));
            return Simd<4, f64>(_mm_xor_ps(a.v128[0], ones), _mm_xor_ps(a.v128[1], ones));
#endif
        }
        static inline Simd<4, f64> landnot(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_andnot_pd(b.v128[0], a.v128[0]), _mm_andnot_pd(b.v128[1], a.v128[1]));
#else
            return Simd<4, f64>(
                _mm_andnot_ps(b.v128[0], a.v128[0]), _mm_andnot_ps(b.v128[1], a.v128[1]));
#endif
        }

        static inline Simd<4, f64> mul(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_mul_pd(a.v128[0], b.v128[0]), _mm_mul_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(a.f[0] * b.f[0], a.f[1] * b.f[1], a.f[2] * b.f[2], a.f[3] * b.f[3]);
#endif
        }
        static inline Simd<4, f64> div(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_div_pd(a.v128[0], b.v128[0]), _mm_div_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(a.f[0] / b.f[0], a.f[1] / b.f[1], a.f[2] / b.f[2], a.f[3] / b.f[3]);
#endif
        }
        static inline Simd<4, f64> sqrt(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(_mm_sqrt_pd(a.v128[0]), _mm_sqrt_pd(a.v128[1]));
#else
            return Simd<4, f64>(::sqrt(a.f[0]), ::sqrt(a.f[1]), ::sqrt(a.f[2]), ::sqrt(a.f[3]));
#endif
        }
        static inline Simd<4, f64> rsqrt(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<4, f64>(
                _mm_div_pd(one, _mm_sqrt_pd(a.v128[0])), _mm_div_pd(one, _mm_sqrt_pd(a.v128[1])));
#else
            return Simd<4, f64>(
                1.0 / ::sqrt(a.f[0]),
                1.0 / ::sqrt(a.f[1]),
                1.0 / ::sqrt(a.f[2]),
                1.0 / ::sqrt(a.f[3]));
#endif
        }
        static inline Simd<4, f64> reciprocal(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<4, f64>(_mm_div_pd(one, a.v128[0]), _mm_div_pd(one, a.v128[1]));
#else
            return Simd<4, f64>(1.0 / a.f[0], 1.0 / a.f[1], 1.0 / a.f[2], 1.0 / a.f[3]);
#endif
        }

        static inline Simd<4, f64> cmpEq(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmpeq_pd(a.v128[0], b.v128[0]), _mm_cmpeq_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] == b.f[0] ? ~0 : 0,
                a.f[1] == b.f[1] ? ~0 : 0,
                a.f[2] == b.f[2] ? ~0 : 0,
                a.f[3] == b.f[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, f64> cmpNe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmpneq_pd(a.v128[0], b.v128[0]), _mm_cmpneq_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] != b.f[0] ? ~0 : 0,
                a.f[1] != b.f[1] ? ~0 : 0,
                a.f[2] != b.f[2] ? ~0 : 0,
                a.f[3] != b.f[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, f64> cmpLt(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmplt_pd(b.v128[0], a.v128[0]), _mm_cmplt_pd(b.v128[1], a.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] < b.f[0] ? ~0 : 0,
                a.f[1] < b.f[1] ? ~0 : 0,
                a.f[2] < b.f[2] ? ~0 : 0,
                a.f[3] < b.f[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, f64> cmpLe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmple_pd(b.v128[0], a.v128[0]), _mm_cmple_pd(b.v128[1], a.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] <= b.f[0] ? ~0 : 0,
                a.f[1] <= b.f[1] ? ~0 : 0,
                a.f[2] <= b.f[2] ? ~0 : 0,
                a.f[3] <= b.f[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, f64> cmpGt(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmpgt_pd(a.v128[0], b.v128[0]), _mm_cmpgt_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] > b.f[0] ? ~0 : 0,
                a.f[1] > b.f[1] ? ~0 : 0,
                a.f[2] > b.f[2] ? ~0 : 0,
                a.f[3] > b.f[3] ? ~0 : 0);
#endif
        }
        static inline Simd<4, f64> cmpGe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<4, f64>(
                _mm_cmpge_pd(a.v128[0], b.v128[0]), _mm_cmpge_pd(a.v128[1], b.v128[1]));
#else
            return Simd<4, f64>(
                a.f[0] >= b.f[0] ? ~0 : 0,
                a.f[1] >= b.f[1] ? ~0 : 0,
                a.f[2] >= b.f[2] ? ~0 : 0,
                a.f[3] >= b.f[3] ? ~0 : 0);
#endif
        }

        static inline u32 movemask(const Simd<4, f64> &a) {
#if ATOM_SIMD_SSE2
            return (u32)_mm_movemask_pd(a.v128[0]) | ((u32)_mm_movemask_pd(a.v128[1]) << 16);
#else
            return (u32)_mm_movemask_ps(a.v128[0]) | ((u32)_mm_movemask_ps(a.v128[1]) << 16);
#endif
        }
    };

    template <>
    union Simd<8, f64> {
#if ATOM_SIMD_SSE2
        __m128d v128[4];
#else
        __m128 v128[4];
#endif
        f64 f[8];
        Simd() {}
#if ATOM_SIMD_SSE2
        Simd(__m128d a0, __m128d a1, __m128d a2, __m128d a3) {
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
        Simd(f64 a0, f64 a1, f64 a2, f64 a3, f64 a4, f64 a5, f64 a6, f64 a7) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
            f[4] = a4;
            f[5] = a5;
            f[6] = a6;
            f[7] = a7;
        }

        static inline Simd<8, f64> load(const f64* ptr) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_loadu_pd(ptr),
                _mm_loadu_pd(ptr + 2),
                _mm_loadu_pd(ptr + 4),
                _mm_loadu_pd(ptr + 6));
#else
            return Simd<8, f64>(
                _mm_loadu_ps((f32*)(ptr)),
                _mm_loadu_ps((f32*)(ptr + 2)),
                _mm_loadu_ps((f32*)(ptr + 4)),
                _mm_loadu_ps((f32*)(ptr + 6)));
#endif
        }
        static inline void store(f64* ptr, const Simd<8, f64> &simd) {
#if ATOM_SIMD_SSE2
            _mm_storeu_pd(ptr, simd.v128[0]);
            _mm_storeu_pd(ptr + 2, simd.v128[1]);
            _mm_storeu_pd(ptr + 4, simd.v128[2]);
            _mm_storeu_pd(ptr + 6, simd.v128[3]);
#else
            _mm_storeu_ps((f32*)(ptr), simd.v128[0]);
            _mm_storeu_ps((f32*)(ptr + 2), simd.v128[1]);
            _mm_storeu_ps((f32*)(ptr + 4), simd.v128[2]);
            _mm_storeu_ps((f32*)(ptr + 6), simd.v128[3]);
#endif
        }
        static inline Simd<8, f64> set1(const f64 value) {
#if ATOM_SIMD_SSE2
            __m128d v = _mm_set1_pd(value);
            return Simd<8, f64>(v, v, v, v);
#else
            __m128 v = _mm_set_ps(
                (f32)((u64)value >> 32), (f32)value, (f32)((u64)value >> 32), (f32)value);
            return Simd<8, f64>(v, v, v, v);
#endif
        }
        static inline Simd<8, f64>
            set(const f64 a0,
                const f64 a1,
                const f64 a2,
                const f64 a3,
                const f64 a4,
                const f64 a5,
                const f64 a6,
                const f64 a7) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_set_pd(a1, a0), _mm_set_pd(a3, a2), _mm_set_pd(a5, a4), _mm_set_pd(a7, a6));
#else
            return Simd<8, f64>(
                _mm_set_ps((f32)((u64)a1 >> 32), (f32)a1, (f32)((u64)a0 >> 32), (f32)a0),
                _mm_set_ps((f32)((u64)a3 >> 32), (f32)a3, (f32)((u64)a2 >> 32), (f32)a2),
                _mm_set_ps((f32)((u64)a5 >> 32), (f32)a5, (f32)((u64)a4 >> 32), (f32)a4),
                _mm_set_ps((f32)((u64)a7 >> 32), (f32)a7, (f32)((u64)a6 >> 32), (f32)a6));
#endif
        }

        static inline Simd<8, f64> add(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_add_pd(a.v128[0], b.v128[0]),
                _mm_add_pd(a.v128[1], b.v128[1]),
                _mm_add_pd(a.v128[2], b.v128[2]),
                _mm_add_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] + b.f[0],
                a.f[1] + b.f[1],
                a.f[2] + b.f[2],
                a.f[3] + b.f[3],
                a.f[4] + b.f[4],
                a.f[5] + b.f[5],
                a.f[6] + b.f[6],
                a.f[7] + b.f[7]);
#endif
        }
        static inline Simd<8, f64> sub(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_sub_pd(a.v128[0], b.v128[0]),
                _mm_sub_pd(a.v128[1], b.v128[1]),
                _mm_sub_pd(a.v128[2], b.v128[2]),
                _mm_sub_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] - b.f[0],
                a.f[1] - b.f[1],
                a.f[2] - b.f[2],
                a.f[3] - b.f[3],
                a.f[4] - b.f[4],
                a.f[5] - b.f[5],
                a.f[6] - b.f[6],
                a.f[7] - b.f[7]);
#endif
        }
        static inline Simd<8, f64> abs(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            __maskd mask = _mm_set1_pd(-0.0);
            return Simd<8, f64>(
                _mm_andnot_pd(mask, a.v128[0]),
                _mm_andnot_pd(mask, a.v128[1]),
                _mm_andnot_pd(mask, a.v128[2]),
                _mm_andnot_pd(mask, a.v128[3]));
#else
            return Simd<8, f64>(
                fabs(a.f[0]),
                fabs(a.f[1]),
                fabs(a.f[2]),
                fabs(a.f[3]),
                fabs(a.f[4]),
                fabs(a.f[5]),
                fabs(a.f[6]),
                fabs(a.f[7]));
#endif
        }
        static inline Simd<8, f64> neg(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d sign_bit = _mm_set1_pd(-0.0);
            return Simd<8, f64>(
                _mm_xor_pd(a.v128[0], sign_bit),
                _mm_xor_pd(a.v128[1], sign_bit),
                _mm_xor_pd(a.v128[2], sign_bit),
                _mm_xor_pd(a.v128[3], sign_bit));
#else
            return Simd<8, f64>(
                -a.f[0], -a.f[1], -a.f[2], -a.f[3], -a.f[4], -a.f[5], -a.f[6], -a.f[7]);
#endif
        }
        static inline Simd<8, f64> min(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_min_pd(a.v128[0], b.v128[0]),
                _mm_min_pd(a.v128[1], b.v128[1]),
                _mm_min_pd(a.v128[2], b.v128[2]),
                _mm_min_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3],
                a.f[4] < b.f[4] ? a.f[4] : b.f[4],
                a.f[5] < b.f[5] ? a.f[5] : b.f[5],
                a.f[6] < b.f[6] ? a.f[6] : b.f[6],
                a.f[7] < b.f[7] ? a.f[7] : b.f[7]);
#endif
        }
        static inline Simd<8, f64> max(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_max_pd(a.v128[0], b.v128[0]),
                _mm_max_pd(a.v128[1], b.v128[1]),
                _mm_max_pd(a.v128[2], b.v128[2]),
                _mm_max_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3],
                a.f[4] > b.f[4] ? a.f[4] : b.f[4],
                a.f[5] > b.f[5] ? a.f[5] : b.f[5],
                a.f[6] > b.f[6] ? a.f[6] : b.f[6],
                a.f[7] > b.f[7] ? a.f[7] : b.f[7]);
#endif
        }

        static inline Simd<8, f64> land(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_and_pd(a.v128[0], b.v128[0]),
                _mm_and_pd(a.v128[1], b.v128[1]),
                _mm_and_pd(a.v128[2], b.v128[2]),
                _mm_and_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                _mm_and_ps(a.v128[0], b.v128[0]),
                _mm_and_ps(a.v128[1], b.v128[1]),
                _mm_and_ps(a.v128[2], b.v128[2]),
                _mm_and_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<8, f64> lor(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_or_pd(a.v128[0], b.v128[0]),
                _mm_or_pd(a.v128[1], b.v128[1]),
                _mm_or_pd(a.v128[2], b.v128[2]),
                _mm_or_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                _mm_or_ps(a.v128[0], b.v128[0]),
                _mm_or_ps(a.v128[1], b.v128[1]),
                _mm_or_ps(a.v128[2], b.v128[2]),
                _mm_or_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<8, f64> lxor(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_xor_pd(a.v128[0], b.v128[0]),
                _mm_xor_pd(a.v128[1], b.v128[1]),
                _mm_xor_pd(a.v128[2], b.v128[2]),
                _mm_xor_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                _mm_xor_ps(a.v128[0], b.v128[0]),
                _mm_xor_ps(a.v128[1], b.v128[1]),
                _mm_xor_ps(a.v128[2], b.v128[2]),
                _mm_xor_ps(a.v128[3], b.v128[3]));
#endif
        }
        static inline Simd<8, f64> lnot(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d ones = _mm_castsi128_pd(_mm_set1_epi64x(~0));
            return Simd<8, f64>(
                _mm_xor_pd(a.v128[0], ones),
                _mm_xor_pd(a.v128[1], ones),
                _mm_xor_pd(a.v128[2], ones),
                _mm_xor_pd(a.v128[3], ones));
#else
            __m128 ones = _mm_castsi128_ps(_mm_set1_epi32(~0));
            return Simd<8, f64>(
                _mm_xor_ps(a.v128[0], ones),
                _mm_xor_ps(a.v128[1], ones),
                _mm_xor_ps(a.v128[2], ones),
                _mm_xor_ps(a.v128[3], ones));
#endif
        }
        static inline Simd<8, f64> landnot(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_andnot_pd(b.v128[0], a.v128[0]),
                _mm_andnot_pd(b.v128[1], a.v128[1]),
                _mm_andnot_pd(b.v128[2], a.v128[2]),
                _mm_andnot_pd(b.v128[3], a.v128[3]));
#else
            return Simd<8, f64>(
                _mm_andnot_ps(b.v128[0], a.v128[0]),
                _mm_andnot_ps(b.v128[1], a.v128[1]),
                _mm_andnot_ps(b.v128[2], a.v128[2]),
                _mm_andnot_ps(b.v128[3], a.v128[3]));
#endif
        }

        static inline Simd<8, f64> mul(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_mul_pd(a.v128[0], b.v128[0]),
                _mm_mul_pd(a.v128[1], b.v128[1]),
                _mm_mul_pd(a.v128[2], b.v128[2]),
                _mm_mul_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] * b.f[0],
                a.f[1] * b.f[1],
                a.f[2] * b.f[2],
                a.f[3] * b.f[3],
                a.f[4] * b.f[4],
                a.f[5] * b.f[5],
                a.f[6] * b.f[6],
                a.f[7] * b.f[7]);
#endif
        }
        static inline Simd<8, f64> div(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_div_pd(a.v128[0], b.v128[0]),
                _mm_div_pd(a.v128[1], b.v128[1]),
                _mm_div_pd(a.v128[2], b.v128[2]),
                _mm_div_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] / b.f[0],
                a.f[1] / b.f[1],
                a.f[2] / b.f[2],
                a.f[3] / b.f[3],
                a.f[4] / b.f[4],
                a.f[5] / b.f[5],
                a.f[6] / b.f[6],
                a.f[7] / b.f[7]);
#endif
        }
        static inline Simd<8, f64> sqrt(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_sqrt_pd(a.v128[0]),
                _mm_sqrt_pd(a.v128[1]),
                _mm_sqrt_pd(a.v128[2]),
                _mm_sqrt_pd(a.v128[3]));
#else
            return Simd<8, f64>(
                ::sqrt(a.f[0]),
                ::sqrt(a.f[1]),
                ::sqrt(a.f[2]),
                ::sqrt(a.f[3]),
                ::sqrt(a.f[4]),
                ::sqrt(a.f[5]),
                ::sqrt(a.f[6]),
                ::sqrt(a.f[7]));
#endif
        }
        static inline Simd<8, f64> rsqrt(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<8, f64>(
                _mm_div_pd(one, _mm_sqrt_pd(a.v128[0])),
                _mm_div_pd(one, _mm_sqrt_pd(a.v128[1])),
                _mm_div_pd(one, _mm_sqrt_pd(a.v128[2])),
                _mm_div_pd(one, _mm_sqrt_pd(a.v128[3])));
#else
            return Simd<8, f64>(
                1.0 / ::sqrt(a.f[0]),
                1.0 / ::sqrt(a.f[1]),
                1.0 / ::sqrt(a.f[2]),
                1.0 / ::sqrt(a.f[3]),
                1.0 / ::sqrt(a.f[4]),
                1.0 / ::sqrt(a.f[5]),
                1.0 / ::sqrt(a.f[6]),
                1.0 / ::sqrt(a.f[7]));
#endif
        }
        static inline Simd<8, f64> reciprocal(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            __m128d one = _mm_set1_pd(1.0);
            return Simd<8, f64>(
                _mm_div_pd(one, a.v128[0]),
                _mm_div_pd(one, a.v128[1]),
                _mm_div_pd(one, a.v128[2]),
                _mm_div_pd(one, a.v128[3]));
#else
            return Simd<8, f64>(
                1.0 / a.f[0],
                1.0 / a.f[1],
                1.0 / a.f[2],
                1.0 / a.f[3],
                1.0 / a.f[4],
                1.0 / a.f[5],
                1.0 / a.f[6],
                1.0 / a.f[7]);
#endif
        }

        static inline Simd<8, f64> cmpEq(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmpeq_pd(a.v128[0], b.v128[0]),
                _mm_cmpeq_pd(a.v128[1], b.v128[1]),
                _mm_cmpeq_pd(a.v128[2], b.v128[2]),
                _mm_cmpeq_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] == b.f[0] ? ~0 : 0,
                a.f[1] == b.f[1] ? ~0 : 0,
                a.f[2] == b.f[2] ? ~0 : 0,
                a.f[3] == b.f[3] ? ~0 : 0,
                a.f[4] == b.f[4] ? ~0 : 0,
                a.f[5] == b.f[5] ? ~0 : 0,
                a.f[6] == b.f[6] ? ~0 : 0,
                a.f[7] == b.f[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, f64> cmpNe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmpneq_pd(a.v128[0], b.v128[0]),
                _mm_cmpneq_pd(a.v128[1], b.v128[1]),
                _mm_cmpneq_pd(a.v128[2], b.v128[2]),
                _mm_cmpneq_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] != b.f[0] ? ~0 : 0,
                a.f[1] != b.f[1] ? ~0 : 0,
                a.f[2] != b.f[2] ? ~0 : 0,
                a.f[3] != b.f[3] ? ~0 : 0,
                a.f[4] != b.f[4] ? ~0 : 0,
                a.f[5] != b.f[5] ? ~0 : 0,
                a.f[6] != b.f[6] ? ~0 : 0,
                a.f[7] != b.f[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, f64> cmpLt(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmplt_pd(b.v128[0], a.v128[0]),
                _mm_cmplt_pd(b.v128[1], a.v128[1]),
                _mm_cmplt_pd(b.v128[2], a.v128[2]),
                _mm_cmplt_pd(b.v128[3], a.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] < b.f[0] ? ~0 : 0,
                a.f[1] < b.f[1] ? ~0 : 0,
                a.f[2] < b.f[2] ? ~0 : 0,
                a.f[3] < b.f[3] ? ~0 : 0,
                a.f[4] < b.f[4] ? ~0 : 0,
                a.f[5] < b.f[5] ? ~0 : 0,
                a.f[6] < b.f[6] ? ~0 : 0,
                a.f[7] < b.f[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, f64> cmpLe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmple_pd(b.v128[0], a.v128[0]),
                _mm_cmple_pd(b.v128[1], a.v128[1]),
                _mm_cmple_pd(b.v128[2], a.v128[2]),
                _mm_cmple_pd(b.v128[3], a.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] <= b.f[0] ? ~0 : 0,
                a.f[1] <= b.f[1] ? ~0 : 0,
                a.f[2] <= b.f[2] ? ~0 : 0,
                a.f[3] <= b.f[3] ? ~0 : 0,
                a.f[4] <= b.f[4] ? ~0 : 0,
                a.f[5] <= b.f[5] ? ~0 : 0,
                a.f[6] <= b.f[6] ? ~0 : 0,
                a.f[7] <= b.f[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, f64> cmpGt(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmpgt_pd(a.v128[0], b.v128[0]),
                _mm_cmpgt_pd(a.v128[1], b.v128[1]),
                _mm_cmpgt_pd(a.v128[2], b.v128[2]),
                _mm_cmpgt_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] > b.f[0] ? ~0 : 0,
                a.f[1] > b.f[1] ? ~0 : 0,
                a.f[2] > b.f[2] ? ~0 : 0,
                a.f[3] > b.f[3] ? ~0 : 0,
                a.f[4] > b.f[4] ? ~0 : 0,
                a.f[5] > b.f[5] ? ~0 : 0,
                a.f[6] > b.f[6] ? ~0 : 0,
                a.f[7] > b.f[7] ? ~0 : 0);
#endif
        }
        static inline Simd<8, f64> cmpGe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
#if ATOM_SIMD_SSE2
            return Simd<8, f64>(
                _mm_cmpge_pd(a.v128[0], b.v128[0]),
                _mm_cmpge_pd(a.v128[1], b.v128[1]),
                _mm_cmpge_pd(a.v128[2], b.v128[2]),
                _mm_cmpge_pd(a.v128[3], b.v128[3]));
#else
            return Simd<8, f64>(
                a.f[0] >= b.f[0] ? ~0 : 0,
                a.f[1] >= b.f[1] ? ~0 : 0,
                a.f[2] >= b.f[2] ? ~0 : 0,
                a.f[3] >= b.f[3] ? ~0 : 0,
                a.f[4] >= b.f[4] ? ~0 : 0,
                a.f[5] >= b.f[5] ? ~0 : 0,
                a.f[6] >= b.f[6] ? ~0 : 0,
                a.f[7] >= b.f[7] ? ~0 : 0);
#endif
        }

        static inline u64 movemask(const Simd<8, f64> &a) {
#if ATOM_SIMD_SSE2
            return (u64)_mm_movemask_pd(a.v128[0])
                 | ((u64)_mm_movemask_pd(a.v128[1]) << 16)
                 | ((u64)_mm_movemask_pd(a.v128[2]) << 32)
                 | ((u64)_mm_movemask_pd(a.v128[3]) << 48);
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
