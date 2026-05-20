#ifndef ATOM_PRTN_SIMD_GENERIC_F64_H
#define ATOM_PRTN_SIMD_GENERIC_F64_H

/**
 * @file SIMD/Generic/f64.h
 * @brief Generic (scalar fallback) implementations for 64-bit float SIMD.
 *
 * Provides `Proton::Simd<N,f64>` specializations and common vector
 * operations (load/store, arithmetic, bitwise, comparisons and helpers).
 */

#include <math.h>

#include "../Simd.h"
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<2, f64> {
        f64 f[2];
        Simd() {}
        Simd(f64 a0, f64 a1) {
            f[0] = a0;
            f[1] = a1;
        }

        static inline Simd<2, f64> load(const f64* ptr) { return Simd<2, f64>(ptr[0], ptr[1]); }
        static inline void         store(f64* ptr, const Simd<2, f64> &simd) {
            ptr[0] = simd.f[0];
            ptr[1] = simd.f[1];
        }
        static inline Simd<2, f64> set1(const f64 value) { return Simd<2, f64>(value, value); }
        static inline Simd<2, f64> set(const f64 a0, const f64 a1) { return Simd<2, f64>(a0, a1); }

        static inline Simd<2, f64> add(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(a.f[0] + b.f[0], a.f[1] + b.f[1]);
        }
        static inline Simd<2, f64> sub(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(a.f[0] - b.f[0], a.f[1] - b.f[1]);
        }
        static inline Simd<2, f64> abs(const Simd<2, f64> &a) {
            return Simd<2, f64>(a.f[0] < 0 ? -a.f[0] : a.f[0], a.f[1] < 0 ? -a.f[1] : a.f[1]);
        }
        static inline Simd<2, f64> neg(const Simd<2, f64> &a) {
            return Simd<2, f64>(-a.f[0], -a.f[1]);
        }
        static inline Simd<2, f64> min(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0], a.f[1] < b.f[1] ? a.f[1] : b.f[1]);
        }
        static inline Simd<2, f64> max(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0], a.f[1] > b.f[1] ? a.f[1] : b.f[1]);
        }

        static inline Simd<2, f64> land(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];

            i64 result0 = ai0 & bi0;
            i64 result1 = ai1 & bi1;

            return Simd<2, f64>(*(f64*)&result0, *(f64*)&result1);
        }
        static inline Simd<2, f64> lor(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];

            i64 result0 = ai0 | bi0;
            i64 result1 = ai1 | bi1;

            return Simd<2, f64>(*(f64*)&result0, *(f64*)&result1);
        }
        static inline Simd<2, f64> lxor(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];

            i64 result0 = ai0 ^ bi0;
            i64 result1 = ai1 ^ bi1;

            return Simd<2, f64>(*(f64*)&result0, *(f64*)&result1);
        }
        static inline Simd<2, f64> lnot(const Simd<2, f64> &a) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];

            i64 result0 = ~ai0;
            i64 result1 = ~ai1;

            return Simd<2, f64>(*(f64*)&result0, *(f64*)&result1);
        }
        static inline Simd<2, f64> landnot(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];

            i64 result0 = ai0 & ~bi0;
            i64 result1 = ai1 & ~bi1;

            return Simd<2, f64>(*(f64*)&result0, *(f64*)&result1);
        }

        static inline Simd<2, f64> mul(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(a.f[0] * b.f[0], a.f[1] * b.f[1]);
        }
        static inline Simd<2, f64> div(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            return Simd<2, f64>(a.f[0] / b.f[0], a.f[1] / b.f[1]);
        }
        static inline Simd<2, f64> sqrt(const Simd<2, f64> &a) {
            return Simd<2, f64>(::sqrt(a.f[0]), ::sqrt(a.f[1]));
        }
        static inline Simd<2, f64> rsqrt(const Simd<2, f64> &a) {
            return Simd<2, f64>(1.0 / ::sqrt(a.f[0]), 1.0 / ::sqrt(a.f[1]));
        }
        static inline Simd<2, f64> reciprocal(const Simd<2, f64> &a) {
            return Simd<2, f64>(1.0 / a.f[0], 1.0 / a.f[1]);
        }

        static inline Simd<2, f64> cmpEq(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] == b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] == b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }
        static inline Simd<2, f64> cmpNe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] != b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] != b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }
        static inline Simd<2, f64> cmpLt(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] < b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] < b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }
        static inline Simd<2, f64> cmpLe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] <= b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] <= b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }
        static inline Simd<2, f64> cmpGt(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] > b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] > b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }
        static inline Simd<2, f64> cmpGe(const Simd<2, f64> &a, const Simd<2, f64> &b) {
            i64 r0 = a.f[0] >= b.f[0] ? ~0 : 0;
            i64 r1 = a.f[1] >= b.f[1] ? ~0 : 0;
            return Simd<2, f64>(*(f64*)&r0, *(f64*)&r1);
        }

        static inline u32 movemask(const Simd<2, f64> &a) {
            u64 mask = 0;

            mask |= (((u64)a.f[0] >> 7) & 1) << 0;
            mask |= (((u64)a.f[0] >> 15) & 1) << 1;
            mask |= (((u64)a.f[0] >> 23) & 1) << 2;
            mask |= (((u64)a.f[0] >> 31) & 1) << 3;

            mask |= (((u64)a.f[1] >> 7) & 1) << 4;
            mask |= (((u64)a.f[1] >> 15) & 1) << 5;
            mask |= (((u64)a.f[1] >> 23) & 1) << 6;
            mask |= (((u64)a.f[1] >> 31) & 1) << 7;

            mask |= (((u64)a.f[2] >> 7) & 1) << 8;
            mask |= (((u64)a.f[2] >> 15) & 1) << 9;
            mask |= (((u64)a.f[2] >> 23) & 1) << 10;
            mask |= (((u64)a.f[2] >> 31) & 1) << 11;

            mask |= (((u64)a.f[3] >> 7) & 1) << 12;
            mask |= (((u64)a.f[3] >> 15) & 1) << 13;
            mask |= (((u64)a.f[3] >> 23) & 1) << 14;
            mask |= (((u64)a.f[3] >> 31) & 1) << 15;

            return (u32)mask;
        }
    };

    template <>
    union Simd<4, f64> {
        f64 f[4];
        Simd() {}
        Simd(f64 a0, f64 a1, f64 a2, f64 a3) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
        }

        static inline Simd<4, f64> load(const f64* ptr) {
            return Simd<4, f64>(ptr[0], ptr[1], ptr[2], ptr[3]);
        }
        static inline void store(f64* ptr, const Simd<4, f64> &simd) {
            ptr[0] = simd.f[0];
            ptr[1] = simd.f[1];
            ptr[2] = simd.f[2];
            ptr[3] = simd.f[3];
        }
        static inline Simd<4, f64> set1(const f64 value) {
            return Simd<4, f64>(value, value, value, value);
        }
        static inline Simd<4, f64> set(const f64 a0, const f64 a1, const f64 a2, const f64 a3) {
            return Simd<4, f64>(a0, a1, a2, a3);
        }

        static inline Simd<4, f64> add(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(a.f[0] + b.f[0], a.f[1] + b.f[1], a.f[2] + b.f[2], a.f[3] + b.f[3]);
        }
        static inline Simd<4, f64> sub(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(a.f[0] - b.f[0], a.f[1] - b.f[1], a.f[2] - b.f[2], a.f[3] - b.f[3]);
        }
        static inline Simd<4, f64> abs(const Simd<4, f64> &a) {
            return Simd<4, f64>(::fabs(a.f[0]), ::fabs(a.f[1]), ::fabs(a.f[2]), ::fabs(a.f[3]));
        }
        static inline Simd<4, f64> neg(const Simd<4, f64> &a) {
            return Simd<4, f64>(-a.f[0], -a.f[1], -a.f[2], -a.f[3]);
        }
        static inline Simd<4, f64> min(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3]);
        }
        static inline Simd<4, f64> max(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3]);
        }

        static inline Simd<4, f64> land(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];

            i64 result0 = ai0 & bi0;
            i64 result1 = ai1 & bi1;
            i64 result2 = ai2 & bi2;
            i64 result3 = ai3 & bi3;

            return Simd<4, f64>(*(f64*)&result0, *(f64*)&result1, *(f64*)&result2, *(f64*)&result3);
        }
        static inline Simd<4, f64> lor(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];

            i64 result0 = ai0 | bi0;
            i64 result1 = ai1 | bi1;
            i64 result2 = ai2 | bi2;
            i64 result3 = ai3 | bi3;

            return Simd<4, f64>(*(f64*)&result0, *(f64*)&result1, *(f64*)&result2, *(f64*)&result3);
        }
        static inline Simd<4, f64> lxor(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];

            i64 result0 = ai0 ^ bi0;
            i64 result1 = ai1 ^ bi1;
            i64 result2 = ai2 ^ bi2;
            i64 result3 = ai3 ^ bi3;

            return Simd<4, f64>(*(f64*)&result0, *(f64*)&result1, *(f64*)&result2, *(f64*)&result3);
        }
        static inline Simd<4, f64> lnot(const Simd<4, f64> &a) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];

            i64 result0 = ~ai0;
            i64 result1 = ~ai1;
            i64 result2 = ~ai2;
            i64 result3 = ~ai3;

            return Simd<4, f64>(*(f64*)&result0, *(f64*)&result1, *(f64*)&result2, *(f64*)&result3);
        }
        static inline Simd<4, f64> landnot(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];

            i64 result0 = ai0 & ~bi0;
            i64 result1 = ai1 & ~bi1;
            i64 result2 = ai2 & ~bi2;
            i64 result3 = ai3 & ~bi3;

            return Simd<4, f64>(*(f64*)&result0, *(f64*)&result1, *(f64*)&result2, *(f64*)&result3);
        }

        static inline Simd<4, f64> mul(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(a.f[0] * b.f[0], a.f[1] * b.f[1], a.f[2] * b.f[2], a.f[3] * b.f[3]);
        }
        static inline Simd<4, f64> div(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            return Simd<4, f64>(a.f[0] / b.f[0], a.f[1] / b.f[1], a.f[2] / b.f[2], a.f[3] / b.f[3]);
        }
        static inline Simd<4, f64> sqrt(const Simd<4, f64> &a) {
            return Simd<4, f64>(::sqrt(a.f[0]), ::sqrt(a.f[1]), ::sqrt(a.f[2]), ::sqrt(a.f[3]));
        }
        static inline Simd<4, f64> rsqrt(const Simd<4, f64> &a) {
            return Simd<4, f64>(
                1.0 / ::sqrt(a.f[0]),
                1.0 / ::sqrt(a.f[1]),
                1.0 / ::sqrt(a.f[2]),
                1.0 / ::sqrt(a.f[3]));
        }
        static inline Simd<4, f64> reciprocal(const Simd<4, f64> &a) {
            return Simd<4, f64>(1.0 / a.f[0], 1.0 / a.f[1], 1.0 / a.f[2], 1.0 / a.f[3]);
        }

        static inline Simd<4, f64> cmpEq(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] == b.f[0] ? ~0 : 0;
            r[1] = a.f[1] == b.f[1] ? ~0 : 0;
            r[2] = a.f[2] == b.f[2] ? ~0 : 0;
            r[3] = a.f[3] == b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }
        static inline Simd<4, f64> cmpNe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] != b.f[0] ? ~0 : 0;
            r[1] = a.f[1] != b.f[1] ? ~0 : 0;
            r[2] = a.f[2] != b.f[2] ? ~0 : 0;
            r[3] = a.f[3] != b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }
        static inline Simd<4, f64> cmpLt(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] < b.f[0] ? ~0 : 0;
            r[1] = a.f[1] < b.f[1] ? ~0 : 0;
            r[2] = a.f[2] < b.f[2] ? ~0 : 0;
            r[3] = a.f[3] < b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }
        static inline Simd<4, f64> cmpLe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] <= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] <= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] <= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] <= b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }
        static inline Simd<4, f64> cmpGt(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] > b.f[0] ? ~0 : 0;
            r[1] = a.f[1] > b.f[1] ? ~0 : 0;
            r[2] = a.f[2] > b.f[2] ? ~0 : 0;
            r[3] = a.f[3] > b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }
        static inline Simd<4, f64> cmpGe(const Simd<4, f64> &a, const Simd<4, f64> &b) {
            i64 r[4];
            r[0] = a.f[0] >= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] >= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] >= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] >= b.f[3] ? ~0 : 0;
            return Simd<4, f64>(*(f64*)&r[0], *(f64*)&r[1], *(f64*)&r[2], *(f64*)&r[3]);
        }

        static inline u32 movemask(const Simd<4, f64> &a) {
            u64 mask = 0;

            mask |= (((u64)a.f[0] >> 7) & 1) << 0;
            mask |= (((u64)a.f[0] >> 15) & 1) << 1;
            mask |= (((u64)a.f[0] >> 23) & 1) << 2;
            mask |= (((u64)a.f[0] >> 31) & 1) << 3;

            mask |= (((u64)a.f[1] >> 7) & 1) << 4;
            mask |= (((u64)a.f[1] >> 15) & 1) << 5;
            mask |= (((u64)a.f[1] >> 23) & 1) << 6;
            mask |= (((u64)a.f[1] >> 31) & 1) << 7;

            mask |= (((u64)a.f[2] >> 7) & 1) << 8;
            mask |= (((u64)a.f[2] >> 15) & 1) << 9;
            mask |= (((u64)a.f[2] >> 23) & 1) << 10;
            mask |= (((u64)a.f[2] >> 31) & 1) << 11;

            mask |= (((u64)a.f[3] >> 7) & 1) << 12;
            mask |= (((u64)a.f[3] >> 15) & 1) << 13;
            mask |= (((u64)a.f[3] >> 23) & 1) << 14;
            mask |= (((u64)a.f[3] >> 31) & 1) << 15;

            mask |= (((u64)a.f[4] >> 7) & 1) << 16;
            mask |= (((u64)a.f[4] >> 15) & 1) << 17;
            mask |= (((u64)a.f[4] >> 23) & 1) << 18;
            mask |= (((u64)a.f[4] >> 31) & 1) << 19;

            mask |= (((u64)a.f[5] >> 7) & 1) << 20;
            mask |= (((u64)a.f[5] >> 15) & 1) << 21;
            mask |= (((u64)a.f[5] >> 23) & 1) << 22;
            mask |= (((u64)a.f[5] >> 31) & 1) << 23;

            mask |= (((u64)a.f[6] >> 7) & 1) << 24;
            mask |= (((u64)a.f[6] >> 15) & 1) << 25;
            mask |= (((u64)a.f[6] >> 23) & 1) << 26;
            mask |= (((u64)a.f[6] >> 31) & 1) << 27;

            mask |= (((u64)a.f[7] >> 7) & 1) << 28;
            mask |= (((u64)a.f[7] >> 15) & 1) << 29;
            mask |= (((u64)a.f[7] >> 23) & 1) << 30;
            mask |= (((u64)a.f[7] >> 31) & 1) << 31;

            return (u32)mask;
        }
    };

    template <>
    union Simd<8, f64> {
        f64 f[8];
        Simd() {}
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
            return Simd<8, f64>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
        }
        static inline void store(f64* ptr, const Simd<8, f64> &simd) {
            ptr[0] = simd.f[0];
            ptr[1] = simd.f[1];
            ptr[2] = simd.f[2];
            ptr[3] = simd.f[3];
            ptr[4] = simd.f[4];
            ptr[5] = simd.f[5];
            ptr[6] = simd.f[6];
            ptr[7] = simd.f[7];
        }
        static inline Simd<8, f64> set1(const f64 value) {
            return Simd<8, f64>(value, value, value, value, value, value, value, value);
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
            return Simd<8, f64>(a0, a1, a2, a3, a4, a5, a6, a7);
        }

        static inline Simd<8, f64> add(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] + b.f[0],
                a.f[1] + b.f[1],
                a.f[2] + b.f[2],
                a.f[3] + b.f[3],
                a.f[4] + b.f[4],
                a.f[5] + b.f[5],
                a.f[6] + b.f[6],
                a.f[7] + b.f[7]);
        }
        static inline Simd<8, f64> sub(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] - b.f[0],
                a.f[1] - b.f[1],
                a.f[2] - b.f[2],
                a.f[3] - b.f[3],
                a.f[4] - b.f[4],
                a.f[5] - b.f[5],
                a.f[6] - b.f[6],
                a.f[7] - b.f[7]);
        }
        static inline Simd<8, f64> abs(const Simd<8, f64> &a) {
            return Simd<8, f64>(
                ::fabs(a.f[0]),
                ::fabs(a.f[1]),
                ::fabs(a.f[2]),
                ::fabs(a.f[3]),
                ::fabs(a.f[4]),
                ::fabs(a.f[5]),
                ::fabs(a.f[6]),
                ::fabs(a.f[7]));
        }
        static inline Simd<8, f64> neg(const Simd<8, f64> &a) {
            return Simd<8, f64>(
                -a.f[0], -a.f[1], -a.f[2], -a.f[3], -a.f[4], -a.f[5], -a.f[6], -a.f[7]);
        }
        static inline Simd<8, f64> min(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3],
                a.f[4] < b.f[4] ? a.f[4] : b.f[4],
                a.f[5] < b.f[5] ? a.f[5] : b.f[5],
                a.f[6] < b.f[6] ? a.f[6] : b.f[6],
                a.f[7] < b.f[7] ? a.f[7] : b.f[7]);
        }
        static inline Simd<8, f64> max(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3],
                a.f[4] > b.f[4] ? a.f[4] : b.f[4],
                a.f[5] > b.f[5] ? a.f[5] : b.f[5],
                a.f[6] > b.f[6] ? a.f[6] : b.f[6],
                a.f[7] > b.f[7] ? a.f[7] : b.f[7]);
        }

        static inline Simd<8, f64> land(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 ai4 = *(i64*)&a.f[4];
            i64 ai5 = *(i64*)&a.f[5];
            i64 ai6 = *(i64*)&a.f[6];
            i64 ai7 = *(i64*)&a.f[7];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];
            i64 bi4 = *(i64*)&b.f[4];
            i64 bi5 = *(i64*)&b.f[5];
            i64 bi6 = *(i64*)&b.f[6];
            i64 bi7 = *(i64*)&b.f[7];

            i64 result0 = ai0 & bi0;
            i64 result1 = ai1 & bi1;
            i64 result2 = ai2 & bi2;
            i64 result3 = ai3 & bi3;
            i64 result4 = ai4 & bi4;
            i64 result5 = ai5 & bi5;
            i64 result6 = ai6 & bi6;
            i64 result7 = ai7 & bi7;

            return Simd<8, f64>(
                *(f64*)&result0,
                *(f64*)&result1,
                *(f64*)&result2,
                *(f64*)&result3,
                *(f64*)&result4,
                *(f64*)&result5,
                *(f64*)&result6,
                *(f64*)&result7);
        }
        static inline Simd<8, f64> lor(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 ai4 = *(i64*)&a.f[4];
            i64 ai5 = *(i64*)&a.f[5];
            i64 ai6 = *(i64*)&a.f[6];
            i64 ai7 = *(i64*)&a.f[7];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];
            i64 bi4 = *(i64*)&b.f[4];
            i64 bi5 = *(i64*)&b.f[5];
            i64 bi6 = *(i64*)&b.f[6];
            i64 bi7 = *(i64*)&b.f[7];

            i64 result0 = ai0 | bi0;
            i64 result1 = ai1 | bi1;
            i64 result2 = ai2 | bi2;
            i64 result3 = ai3 | bi3;
            i64 result4 = ai4 | bi4;
            i64 result5 = ai5 | bi5;
            i64 result6 = ai6 | bi6;
            i64 result7 = ai7 | bi7;

            return Simd<8, f64>(
                *(f64*)&result0,
                *(f64*)&result1,
                *(f64*)&result2,
                *(f64*)&result3,
                *(f64*)&result4,
                *(f64*)&result5,
                *(f64*)&result6,
                *(f64*)&result7);
        }
        static inline Simd<8, f64> lxor(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 ai4 = *(i64*)&a.f[4];
            i64 ai5 = *(i64*)&a.f[5];
            i64 ai6 = *(i64*)&a.f[6];
            i64 ai7 = *(i64*)&a.f[7];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];
            i64 bi4 = *(i64*)&b.f[4];
            i64 bi5 = *(i64*)&b.f[5];
            i64 bi6 = *(i64*)&b.f[6];
            i64 bi7 = *(i64*)&b.f[7];

            i64 result0 = ai0 ^ bi0;
            i64 result1 = ai1 ^ bi1;
            i64 result2 = ai2 ^ bi2;
            i64 result3 = ai3 ^ bi3;
            i64 result4 = ai4 ^ bi4;
            i64 result5 = ai5 ^ bi5;
            i64 result6 = ai6 ^ bi6;
            i64 result7 = ai7 ^ bi7;

            return Simd<8, f64>(
                *(f64*)&result0,
                *(f64*)&result1,
                *(f64*)&result2,
                *(f64*)&result3,
                *(f64*)&result4,
                *(f64*)&result5,
                *(f64*)&result6,
                *(f64*)&result7);
        }
        static inline Simd<8, f64> lnot(const Simd<8, f64> &a) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 ai4 = *(i64*)&a.f[4];
            i64 ai5 = *(i64*)&a.f[5];
            i64 ai6 = *(i64*)&a.f[6];
            i64 ai7 = *(i64*)&a.f[7];

            i64 result0 = ~ai0;
            i64 result1 = ~ai1;
            i64 result2 = ~ai2;
            i64 result3 = ~ai3;
            i64 result4 = ~ai4;
            i64 result5 = ~ai5;
            i64 result6 = ~ai6;
            i64 result7 = ~ai7;

            return Simd<8, f64>(
                *(f64*)&result0,
                *(f64*)&result1,
                *(f64*)&result2,
                *(f64*)&result3,
                *(f64*)&result4,
                *(f64*)&result5,
                *(f64*)&result6,
                *(f64*)&result7);
        }
        static inline Simd<8, f64> landnot(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 ai0 = *(i64*)&a.f[0];
            i64 ai1 = *(i64*)&a.f[1];
            i64 ai2 = *(i64*)&a.f[2];
            i64 ai3 = *(i64*)&a.f[3];
            i64 ai4 = *(i64*)&a.f[4];
            i64 ai5 = *(i64*)&a.f[5];
            i64 ai6 = *(i64*)&a.f[6];
            i64 ai7 = *(i64*)&a.f[7];
            i64 bi0 = *(i64*)&b.f[0];
            i64 bi1 = *(i64*)&b.f[1];
            i64 bi2 = *(i64*)&b.f[2];
            i64 bi3 = *(i64*)&b.f[3];
            i64 bi4 = *(i64*)&b.f[4];
            i64 bi5 = *(i64*)&b.f[5];
            i64 bi6 = *(i64*)&b.f[6];
            i64 bi7 = *(i64*)&b.f[7];

            i64 result0 = ai0 & ~bi0;
            i64 result1 = ai1 & ~bi1;
            i64 result2 = ai2 & ~bi2;
            i64 result3 = ai3 & ~bi3;
            i64 result4 = ai4 & ~bi4;
            i64 result5 = ai5 & ~bi5;
            i64 result6 = ai6 & ~bi6;
            i64 result7 = ai7 & ~bi7;

            return Simd<8, f64>(
                *(f64*)&result0,
                *(f64*)&result1,
                *(f64*)&result2,
                *(f64*)&result3,
                *(f64*)&result4,
                *(f64*)&result5,
                *(f64*)&result6,
                *(f64*)&result7);
        }

        static inline Simd<8, f64> mul(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] * b.f[0],
                a.f[1] * b.f[1],
                a.f[2] * b.f[2],
                a.f[3] * b.f[3],
                a.f[4] * b.f[4],
                a.f[5] * b.f[5],
                a.f[6] * b.f[6],
                a.f[7] * b.f[7]);
        }
        static inline Simd<8, f64> div(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            return Simd<8, f64>(
                a.f[0] / b.f[0],
                a.f[1] / b.f[1],
                a.f[2] / b.f[2],
                a.f[3] / b.f[3],
                a.f[4] / b.f[4],
                a.f[5] / b.f[5],
                a.f[6] / b.f[6],
                a.f[7] / b.f[7]);
        }
        static inline Simd<8, f64> sqrt(const Simd<8, f64> &a) {
            return Simd<8, f64>(
                ::sqrt(a.f[0]),
                ::sqrt(a.f[1]),
                ::sqrt(a.f[2]),
                ::sqrt(a.f[3]),
                ::sqrt(a.f[4]),
                ::sqrt(a.f[5]),
                ::sqrt(a.f[6]),
                ::sqrt(a.f[7]));
        }
        static inline Simd<8, f64> rsqrt(const Simd<8, f64> &a) {
            return Simd<8, f64>(
                1.0 / ::sqrt(a.f[0]),
                1.0 / ::sqrt(a.f[1]),
                1.0 / ::sqrt(a.f[2]),
                1.0 / ::sqrt(a.f[3]),
                1.0 / ::sqrt(a.f[4]),
                1.0 / ::sqrt(a.f[5]),
                1.0 / ::sqrt(a.f[6]),
                1.0 / ::sqrt(a.f[7]));
        }
        static inline Simd<8, f64> reciprocal(const Simd<8, f64> &a) {
            return Simd<8, f64>(
                1.0 / a.f[0],
                1.0 / a.f[1],
                1.0 / a.f[2],
                1.0 / a.f[3],
                1.0 / a.f[4],
                1.0 / a.f[5],
                1.0 / a.f[6],
                1.0 / a.f[7]);
        }

        static inline Simd<8, f64> cmpEq(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] == b.f[0] ? ~0 : 0;
            r[1] = a.f[1] == b.f[1] ? ~0 : 0;
            r[2] = a.f[2] == b.f[2] ? ~0 : 0;
            r[3] = a.f[3] == b.f[3] ? ~0 : 0;
            r[4] = a.f[4] == b.f[4] ? ~0 : 0;
            r[5] = a.f[5] == b.f[5] ? ~0 : 0;
            r[6] = a.f[6] == b.f[6] ? ~0 : 0;
            r[7] = a.f[7] == b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }
        static inline Simd<8, f64> cmpNe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] != b.f[0] ? ~0 : 0;
            r[1] = a.f[1] != b.f[1] ? ~0 : 0;
            r[2] = a.f[2] != b.f[2] ? ~0 : 0;
            r[3] = a.f[3] != b.f[3] ? ~0 : 0;
            r[4] = a.f[4] != b.f[4] ? ~0 : 0;
            r[5] = a.f[5] != b.f[5] ? ~0 : 0;
            r[6] = a.f[6] != b.f[6] ? ~0 : 0;
            r[7] = a.f[7] != b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }
        static inline Simd<8, f64> cmpLt(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] < b.f[0] ? ~0 : 0;
            r[1] = a.f[1] < b.f[1] ? ~0 : 0;
            r[2] = a.f[2] < b.f[2] ? ~0 : 0;
            r[3] = a.f[3] < b.f[3] ? ~0 : 0;
            r[4] = a.f[4] < b.f[4] ? ~0 : 0;
            r[5] = a.f[5] < b.f[5] ? ~0 : 0;
            r[6] = a.f[6] < b.f[6] ? ~0 : 0;
            r[7] = a.f[7] < b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }
        static inline Simd<8, f64> cmpLe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] <= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] <= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] <= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] <= b.f[3] ? ~0 : 0;
            r[4] = a.f[4] <= b.f[4] ? ~0 : 0;
            r[5] = a.f[5] <= b.f[5] ? ~0 : 0;
            r[6] = a.f[6] <= b.f[6] ? ~0 : 0;
            r[7] = a.f[7] <= b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }
        static inline Simd<8, f64> cmpGt(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] > b.f[0] ? ~0 : 0;
            r[1] = a.f[1] > b.f[1] ? ~0 : 0;
            r[2] = a.f[2] > b.f[2] ? ~0 : 0;
            r[3] = a.f[3] > b.f[3] ? ~0 : 0;
            r[4] = a.f[4] > b.f[4] ? ~0 : 0;
            r[5] = a.f[5] > b.f[5] ? ~0 : 0;
            r[6] = a.f[6] > b.f[6] ? ~0 : 0;
            r[7] = a.f[7] > b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }
        static inline Simd<8, f64> cmpGe(const Simd<8, f64> &a, const Simd<8, f64> &b) {
            i64 r[8];
            r[0] = a.f[0] >= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] >= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] >= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] >= b.f[3] ? ~0 : 0;
            r[4] = a.f[4] >= b.f[4] ? ~0 : 0;
            r[5] = a.f[5] >= b.f[5] ? ~0 : 0;
            r[6] = a.f[6] >= b.f[6] ? ~0 : 0;
            r[7] = a.f[7] >= b.f[7] ? ~0 : 0;
            return Simd<8, f64>(
                *(f64*)&r[0],
                *(f64*)&r[1],
                *(f64*)&r[2],
                *(f64*)&r[3],
                *(f64*)&r[4],
                *(f64*)&r[5],
                *(f64*)&r[6],
                *(f64*)&r[7]);
        }

        static inline u64 movemask(const Simd<8, f64> &a) {
            u64 mask = 0;

            mask |= (((u64)a.f[0] >> 7) & 1) << 0;
            mask |= (((u64)a.f[0] >> 15) & 1) << 1;
            mask |= (((u64)a.f[0] >> 23) & 1) << 2;
            mask |= (((u64)a.f[0] >> 31) & 1) << 3;

            mask |= (((u64)a.f[1] >> 7) & 1) << 4;
            mask |= (((u64)a.f[1] >> 15) & 1) << 5;
            mask |= (((u64)a.f[1] >> 23) & 1) << 6;
            mask |= (((u64)a.f[1] >> 31) & 1) << 7;

            mask |= (((u64)a.f[2] >> 7) & 1) << 8;
            mask |= (((u64)a.f[2] >> 15) & 1) << 9;
            mask |= (((u64)a.f[2] >> 23) & 1) << 10;
            mask |= (((u64)a.f[2] >> 31) & 1) << 11;

            mask |= (((u64)a.f[3] >> 7) & 1) << 12;
            mask |= (((u64)a.f[3] >> 15) & 1) << 13;
            mask |= (((u64)a.f[3] >> 23) & 1) << 14;
            mask |= (((u64)a.f[3] >> 31) & 1) << 15;

            mask |= (((u64)a.f[4] >> 7) & 1) << 16;
            mask |= (((u64)a.f[4] >> 15) & 1) << 17;
            mask |= (((u64)a.f[4] >> 23) & 1) << 18;
            mask |= (((u64)a.f[4] >> 31) & 1) << 19;

            mask |= (((u64)a.f[5] >> 7) & 1) << 20;
            mask |= (((u64)a.f[5] >> 15) & 1) << 21;
            mask |= (((u64)a.f[5] >> 23) & 1) << 22;
            mask |= (((u64)a.f[5] >> 31) & 1) << 23;

            mask |= (((u64)a.f[6] >> 7) & 1) << 24;
            mask |= (((u64)a.f[6] >> 15) & 1) << 25;
            mask |= (((u64)a.f[6] >> 23) & 1) << 26;
            mask |= (((u64)a.f[6] >> 31) & 1) << 27;

            mask |= (((u64)a.f[7] >> 7) & 1) << 28;
            mask |= (((u64)a.f[7] >> 15) & 1) << 29;
            mask |= (((u64)a.f[7] >> 23) & 1) << 30;
            mask |= (((u64)a.f[7] >> 31) & 1) << 31;

            mask |= (((u64)a.f[8] >> 7) & 1) << 32;
            mask |= (((u64)a.f[8] >> 15) & 1) << 33;
            mask |= (((u64)a.f[8] >> 23) & 1) << 34;
            mask |= (((u64)a.f[8] >> 31) & 1) << 35;

            mask |= (((u64)a.f[9] >> 7) & 1) << 36;
            mask |= (((u64)a.f[9] >> 15) & 1) << 37;
            mask |= (((u64)a.f[9] >> 23) & 1) << 38;
            mask |= (((u64)a.f[9] >> 31) & 1) << 39;

            mask |= (((u64)a.f[10] >> 7) & 1) << 40;
            mask |= (((u64)a.f[10] >> 15) & 1) << 41;
            mask |= (((u64)a.f[10] >> 23) & 1) << 42;
            mask |= (((u64)a.f[10] >> 31) & 1) << 43;

            mask |= (((u64)a.f[11] >> 7) & 1) << 44;
            mask |= (((u64)a.f[11] >> 15) & 1) << 45;
            mask |= (((u64)a.f[11] >> 23) & 1) << 46;
            mask |= (((u64)a.f[11] >> 31) & 1) << 47;

            mask |= (((u64)a.f[12] >> 7) & 1) << 48;
            mask |= (((u64)a.f[12] >> 15) & 1) << 49;
            mask |= (((u64)a.f[12] >> 23) & 1) << 50;
            mask |= (((u64)a.f[12] >> 31) & 1) << 51;

            mask |= (((u64)a.f[13] >> 7) & 1) << 52;
            mask |= (((u64)a.f[13] >> 15) & 1) << 53;
            mask |= (((u64)a.f[13] >> 23) & 1) << 54;
            mask |= (((u64)a.f[13] >> 31) & 1) << 55;

            mask |= (((u64)a.f[14] >> 7) & 1) << 56;
            mask |= (((u64)a.f[14] >> 15) & 1) << 57;
            mask |= (((u64)a.f[14] >> 23) & 1) << 58;
            mask |= (((u64)a.f[14] >> 31) & 1) << 59;

            mask |= (((u64)a.f[15] >> 7) & 1) << 60;
            mask |= (((u64)a.f[15] >> 15) & 1) << 61;
            mask |= (((u64)a.f[15] >> 23) & 1) << 62;
            mask |= (((u64)a.f[15] >> 31) & 1) << 63;

            return mask;
        }
    };
}

#endif
