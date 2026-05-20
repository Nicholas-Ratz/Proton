#ifndef ATOM_PRTN_SIMD_GENERIC_I64_H
#define ATOM_PRTN_SIMD_GENERIC_I64_H

/**
 * @file SIMD/Generic/i64.h
 * @brief Generic (scalar fallback) implementations for 64-bit integer SIMD.
 *
 * Provides `Proton::Simd<2,i64>` specializations and common integer
 * operations (load/store, add/sub, bitwise ops, comparisons).
 */

#include "../Simd.h"
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<2, i64> {
        i64 i[2];
        Simd() {}
        Simd(i64 a0, i64 a1) {
            i[0] = a0;
            i[1] = a1;
        }

        static inline Simd<2, i64> load(const i64* ptr) { return Simd<2, i64>(ptr[0], ptr[1]); }
        static inline void         store(i64* ptr, const Simd<2, i64> &simd) {
            ptr[0] = simd.i[0];
            ptr[1] = simd.i[1];
        }
        static inline Simd<2, i64> set1(const i64 value) { return Simd<2, i64>(value, value); }
        static inline Simd<2, i64> set(const i64 a0, const i64 a1) { return Simd<2, i64>(a0, a1); }

        static inline Simd<2, i64> add(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] + b.i[0], a.i[1] + b.i[1]);
        }
        static inline Simd<2, i64> sub(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] - b.i[0], a.i[1] - b.i[1]);
        }
        static inline Simd<2, i64> abs(const Simd<2, i64> &a) {
            return Simd<2, i64>(a.i[0] < 0 ? -a.i[0] : a.i[0], a.i[1] < 0 ? -a.i[1] : a.i[1]);
        }
        static inline Simd<2, i64> neg(const Simd<2, i64> &a) {
            return Simd<2, i64>(-a.i[0], -a.i[1]);
        }
        static inline Simd<2, i64> min(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0], a.i[1] < b.i[1] ? a.i[1] : b.i[1]);
        }
        static inline Simd<2, i64> max(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0], a.i[1] > b.i[1] ? a.i[1] : b.i[1]);
        }

        static inline Simd<2, i64> land(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] & b.i[0], a.i[1] & b.i[1]);
        }
        static inline Simd<2, i64> lor(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] | b.i[0], a.i[1] | b.i[1]);
        }
        static inline Simd<2, i64> lxor(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] ^ b.i[0], a.i[1] ^ b.i[1]);
        }
        static inline Simd<2, i64> lnot(const Simd<2, i64> &a) {
            return Simd<2, i64>(~a.i[0], ~a.i[1]);
        }
        static inline Simd<2, i64> landnot(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] & ~b.i[0], a.i[1] & ~b.i[1]);
        }

        static inline Simd<2, i64> shl(const Simd<2, i64> &a, const int count) {
            return Simd<2, i64>(a.i[0] << count, a.i[1] << count);
        }
        static inline Simd<2, i64> shr(const Simd<2, i64> &a, const int count) {
            return Simd<2, i64>((u32)a.i[0] >> count, (u32)a.i[1] >> count);
        }
        static inline Simd<2, i64> sar(const Simd<2, i64> &a, const int count) {
            return Simd<2, i64>(a.i[0] >> count, a.i[1] >> count);
        }

        static inline Simd<2, i64> mul(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] * b.i[0], a.i[1] * b.i[1]);
        }
        static inline Simd<2, i64> cmpEq(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] == b.i[0] ? ~0 : 0, a.i[1] == b.i[1] ? ~0 : 0);
        }
        static inline Simd<2, i64> cmpNe(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] != b.i[0] ? ~0 : 0, a.i[1] != b.i[1] ? ~0 : 0);
        }
        static inline Simd<2, i64> cmpLt(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] < b.i[0] ? ~0 : 0, a.i[1] < b.i[1] ? ~0 : 0);
        }
        static inline Simd<2, i64> cmpLe(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] <= b.i[0] ? ~0 : 0, a.i[1] <= b.i[1] ? ~0 : 0);
        }
        static inline Simd<2, i64> cmpGt(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] > b.i[0] ? ~0 : 0, a.i[1] > b.i[1] ? ~0 : 0);
        }
        static inline Simd<2, i64> cmpGe(const Simd<2, i64> &a, const Simd<2, i64> &b) {
            return Simd<2, i64>(a.i[0] >= b.i[0] ? ~0 : 0, a.i[1] >= b.i[1] ? ~0 : 0);
        }

        static inline u32 movemask(const Simd<2, i64> &a) {
            i64 mask = 0;

            mask |= ((a.i[0] >> 7) & 1) << 0;
            mask |= ((a.i[0] >> 15) & 1) << 1;
            mask |= ((a.i[0] >> 23) & 1) << 2;
            mask |= ((a.i[0] >> 31) & 1) << 3;

            mask |= ((a.i[1] >> 7) & 1) << 4;
            mask |= ((a.i[1] >> 15) & 1) << 5;
            mask |= ((a.i[1] >> 23) & 1) << 6;
            mask |= ((a.i[1] >> 31) & 1) << 7;

            mask |= ((a.i[2] >> 7) & 1) << 8;
            mask |= ((a.i[2] >> 15) & 1) << 9;
            mask |= ((a.i[2] >> 23) & 1) << 10;
            mask |= ((a.i[2] >> 31) & 1) << 11;

            mask |= ((a.i[3] >> 7) & 1) << 12;
            mask |= ((a.i[3] >> 15) & 1) << 13;
            mask |= ((a.i[3] >> 23) & 1) << 14;
            mask |= ((a.i[3] >> 31) & 1) << 15;

            return (u32)mask;
        }
    };

    template <>
    union Simd<4, i64> {
        i64 i[4];
        Simd() {}
        Simd(i64 a0, i64 a1, i64 a2, i64 a3) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
        }

        static inline Simd<4, i64> load(const i64* ptr) {
            return Simd<4, i64>(ptr[0], ptr[1], ptr[2], ptr[3]);
        }
        static inline void store(i64* ptr, const Simd<4, i64> &simd) {
            ptr[0] = simd.i[0];
            ptr[1] = simd.i[1];
            ptr[2] = simd.i[2];
            ptr[3] = simd.i[3];
        }
        static inline Simd<4, i64> set1(const i64 value) {
            return Simd<4, i64>(value, value, value, value);
        }
        static inline Simd<4, i64> set(const i64 a0, const i64 a1, const i64 a2, const i64 a3) {
            return Simd<4, i64>(a0, a1, a2, a3);
        }

        static inline Simd<4, i64> add(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] + b.i[0], a.i[1] + b.i[1], a.i[2] + b.i[2], a.i[3] + b.i[3]);
        }
        static inline Simd<4, i64> sub(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] - b.i[0], a.i[1] - b.i[1], a.i[2] - b.i[2], a.i[3] - b.i[3]);
        }
        static inline Simd<4, i64> abs(const Simd<4, i64> &a) {
            return Simd<4, i64>(
                a.i[0] < 0 ? -a.i[0] : a.i[0],
                a.i[1] < 0 ? -a.i[1] : a.i[1],
                a.i[2] < 0 ? -a.i[2] : a.i[2],
                a.i[3] < 0 ? -a.i[3] : a.i[3]);
        }
        static inline Simd<4, i64> neg(const Simd<4, i64> &a) {
            return Simd<4, i64>(-a.i[0], -a.i[1], -a.i[2], -a.i[3]);
        }
        static inline Simd<4, i64> min(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3]);
        }
        static inline Simd<4, i64> max(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3]);
        }

        static inline Simd<4, i64> land(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] & b.i[0], a.i[1] & b.i[1], a.i[2] & b.i[2], a.i[3] & b.i[3]);
        }
        static inline Simd<4, i64> lor(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] | b.i[0], a.i[1] | b.i[1], a.i[2] | b.i[2], a.i[3] | b.i[3]);
        }
        static inline Simd<4, i64> lxor(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] ^ b.i[0], a.i[1] ^ b.i[1], a.i[2] ^ b.i[2], a.i[3] ^ b.i[3]);
        }
        static inline Simd<4, i64> lnot(const Simd<4, i64> &a) {
            return Simd<4, i64>(~a.i[0], ~a.i[1], ~a.i[2], ~a.i[3]);
        }
        static inline Simd<4, i64> landnot(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] & ~b.i[0], a.i[1] & ~b.i[1], a.i[2] & ~b.i[2], a.i[3] & ~b.i[3]);
        }

        static inline Simd<4, i64> shl(const Simd<4, i64> &a, const int count) {
            return Simd<4, i64>(a.i[0] << count, a.i[1] << count, a.i[2] << count, a.i[3] << count);
        }
        static inline Simd<4, i64> shr(const Simd<4, i64> &a, const int count) {
            return Simd<4, i64>(
                (u32)a.i[0] >> count,
                (u32)a.i[1] >> count,
                (u32)a.i[2] >> count,
                (u32)a.i[3] >> count);
        }
        static inline Simd<4, i64> sar(const Simd<4, i64> &a, const int count) {
            return Simd<4, i64>(a.i[0] >> count, a.i[1] >> count, a.i[2] >> count, a.i[3] >> count);
        }
        static inline Simd<4, i64> mul(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(a.i[0] * b.i[0], a.i[1] * b.i[1], a.i[2] * b.i[2], a.i[3] * b.i[3]);
        }

        static inline Simd<4, i64> cmpEq(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i64> cmpNe(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i64> cmpLt(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i64> cmpLe(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i64> cmpGt(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i64> cmpGe(const Simd<4, i64> &a, const Simd<4, i64> &b) {
            return Simd<4, i64>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0);
        }

        static inline u32 movemask(const Simd<4, i64> &a) {
            i64 mask = 0;

            mask |= ((a.i[0] >> 7) & 1) << 0;
            mask |= ((a.i[0] >> 15) & 1) << 1;
            mask |= ((a.i[0] >> 23) & 1) << 2;
            mask |= ((a.i[0] >> 31) & 1) << 3;

            mask |= ((a.i[1] >> 7) & 1) << 4;
            mask |= ((a.i[1] >> 15) & 1) << 5;
            mask |= ((a.i[1] >> 23) & 1) << 6;
            mask |= ((a.i[1] >> 31) & 1) << 7;

            mask |= ((a.i[2] >> 7) & 1) << 8;
            mask |= ((a.i[2] >> 15) & 1) << 9;
            mask |= ((a.i[2] >> 23) & 1) << 10;
            mask |= ((a.i[2] >> 31) & 1) << 11;

            mask |= ((a.i[3] >> 7) & 1) << 12;
            mask |= ((a.i[3] >> 15) & 1) << 13;
            mask |= ((a.i[3] >> 23) & 1) << 14;
            mask |= ((a.i[3] >> 31) & 1) << 15;

            mask |= ((a.i[4] >> 7) & 1) << 16;
            mask |= ((a.i[4] >> 15) & 1) << 17;
            mask |= ((a.i[4] >> 23) & 1) << 18;
            mask |= ((a.i[4] >> 31) & 1) << 19;

            mask |= ((a.i[5] >> 7) & 1) << 20;
            mask |= ((a.i[5] >> 15) & 1) << 21;
            mask |= ((a.i[5] >> 23) & 1) << 22;
            mask |= ((a.i[5] >> 31) & 1) << 23;

            mask |= ((a.i[6] >> 7) & 1) << 24;
            mask |= ((a.i[6] >> 15) & 1) << 25;
            mask |= ((a.i[6] >> 23) & 1) << 26;
            mask |= ((a.i[6] >> 31) & 1) << 27;

            mask |= ((a.i[7] >> 7) & 1) << 28;
            mask |= ((a.i[7] >> 15) & 1) << 29;
            mask |= ((a.i[7] >> 23) & 1) << 30;
            mask |= ((a.i[7] >> 31) & 1) << 31;

            return (u32)mask;
        }
    };

    template <>
    union Simd<8, i64> {
        i64 i[8];
        Simd() {}
        Simd(i64 a0, i64 a1, i64 a2, i64 a3, i64 a4, i64 a5, i64 a6, i64 a7) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
            i[4] = a4;
            i[5] = a5;
            i[6] = a6;
            i[7] = a7;
        }

        static inline Simd<8, i64> load(const i64* ptr) {
            return Simd<8, i64>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
        }
        static inline void store(i64* ptr, const Simd<8, i64> &simd) {
            ptr[0] = simd.i[0];
            ptr[1] = simd.i[1];
            ptr[2] = simd.i[2];
            ptr[3] = simd.i[3];
            ptr[4] = simd.i[4];
            ptr[5] = simd.i[5];
            ptr[6] = simd.i[6];
            ptr[7] = simd.i[7];
        }
        static inline Simd<8, i64> set1(const i64 value) {
            return Simd<8, i64>(value, value, value, value, value, value, value, value);
        }
        static inline Simd<8, i64>
            set(const i64 a0,
                const i64 a1,
                const i64 a2,
                const i64 a3,
                const i64 a4,
                const i64 a5,
                const i64 a6,
                const i64 a7) {
            return Simd<8, i64>(a0, a1, a2, a3, a4, a5, a6, a7);
        }

        static inline Simd<8, i64> add(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] + b.i[0],
                a.i[1] + b.i[1],
                a.i[2] + b.i[2],
                a.i[3] + b.i[3],
                a.i[4] + b.i[4],
                a.i[5] + b.i[5],
                a.i[6] + b.i[6],
                a.i[7] + b.i[7]);
        }
        static inline Simd<8, i64> sub(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] - b.i[0],
                a.i[1] - b.i[1],
                a.i[2] - b.i[2],
                a.i[3] - b.i[3],
                a.i[4] - b.i[4],
                a.i[5] - b.i[5],
                a.i[6] - b.i[6],
                a.i[7] - b.i[7]);
        }
        static inline Simd<8, i64> abs(const Simd<8, i64> &a) {
            return Simd<8, i64>(
                a.i[0] < 0 ? -a.i[0] : a.i[0],
                a.i[1] < 0 ? -a.i[1] : a.i[1],
                a.i[2] < 0 ? -a.i[2] : a.i[2],
                a.i[3] < 0 ? -a.i[3] : a.i[3],
                a.i[4] < 0 ? -a.i[4] : a.i[4],
                a.i[5] < 0 ? -a.i[5] : a.i[5],
                a.i[6] < 0 ? -a.i[6] : a.i[6],
                a.i[7] < 0 ? -a.i[7] : a.i[7]);
        }
        static inline Simd<8, i64> neg(const Simd<8, i64> &a) {
            return Simd<8, i64>(
                -a.i[0], -a.i[1], -a.i[2], -a.i[3], -a.i[4], -a.i[5], -a.i[6], -a.i[7]);
        }
        static inline Simd<8, i64> min(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3],
                a.i[4] < b.i[4] ? a.i[4] : b.i[4],
                a.i[5] < b.i[5] ? a.i[5] : b.i[5],
                a.i[6] < b.i[6] ? a.i[6] : b.i[6],
                a.i[7] < b.i[7] ? a.i[7] : b.i[7]);
        }
        static inline Simd<8, i64> max(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3],
                a.i[4] > b.i[4] ? a.i[4] : b.i[4],
                a.i[5] > b.i[5] ? a.i[5] : b.i[5],
                a.i[6] > b.i[6] ? a.i[6] : b.i[6],
                a.i[7] > b.i[7] ? a.i[7] : b.i[7]);
        }

        static inline Simd<8, i64> land(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] & b.i[0],
                a.i[1] & b.i[1],
                a.i[2] & b.i[2],
                a.i[3] & b.i[3],
                a.i[4] & b.i[4],
                a.i[5] & b.i[5],
                a.i[6] & b.i[6],
                a.i[7] & b.i[7]);
        }
        static inline Simd<8, i64> lor(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] | b.i[0],
                a.i[1] | b.i[1],
                a.i[2] | b.i[2],
                a.i[3] | b.i[3],
                a.i[4] | b.i[4],
                a.i[5] | b.i[5],
                a.i[6] | b.i[6],
                a.i[7] | b.i[7]);
        }
        static inline Simd<8, i64> lxor(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] ^ b.i[0],
                a.i[1] ^ b.i[1],
                a.i[2] ^ b.i[2],
                a.i[3] ^ b.i[3],
                a.i[4] ^ b.i[4],
                a.i[5] ^ b.i[5],
                a.i[6] ^ b.i[6],
                a.i[7] ^ b.i[7]);
        }
        static inline Simd<8, i64> lnot(const Simd<8, i64> &a) {
            return Simd<8, i64>(
                ~a.i[0], ~a.i[1], ~a.i[2], ~a.i[3], ~a.i[4], ~a.i[5], ~a.i[6], ~a.i[7]);
        }
        static inline Simd<8, i64> landnot(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] & ~b.i[0],
                a.i[1] & ~b.i[1],
                a.i[2] & ~b.i[2],
                a.i[3] & ~b.i[3],
                a.i[4] & ~b.i[4],
                a.i[5] & ~b.i[5],
                a.i[6] & ~b.i[6],
                a.i[7] & ~b.i[7]);
        }

        static inline Simd<8, i64> shl(const Simd<8, i64> &a, const int count) {
            return Simd<8, i64>(
                a.i[0] << count,
                a.i[1] << count,
                a.i[2] << count,
                a.i[3] << count,
                a.i[4] << count,
                a.i[5] << count,
                a.i[6] << count,
                a.i[7] << count);
        }
        static inline Simd<8, i64> shr(const Simd<8, i64> &a, const int count) {
            return Simd<8, i64>(
                (u32)a.i[0] >> count,
                (u32)a.i[1] >> count,
                (u32)a.i[2] >> count,
                (u32)a.i[3] >> count,
                (u32)a.i[4] >> count,
                (u32)a.i[5] >> count,
                (u32)a.i[6] >> count,
                (u32)a.i[7] >> count);
        }
        static inline Simd<8, i64> sar(const Simd<8, i64> &a, const int count) {
            return Simd<8, i64>(
                a.i[0] >> count,
                a.i[1] >> count,
                a.i[2] >> count,
                a.i[3] >> count,
                a.i[4] >> count,
                a.i[5] >> count,
                a.i[6] >> count,
                a.i[7] >> count);
        }

        static inline Simd<8, i64> mul(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] * b.i[0],
                a.i[1] * b.i[1],
                a.i[2] * b.i[2],
                a.i[3] * b.i[3],
                a.i[4] * b.i[4],
                a.i[5] * b.i[5],
                a.i[6] * b.i[6],
                a.i[7] * b.i[7]);
        }

        static inline Simd<8, i64> cmpEq(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0,
                a.i[4] == b.i[4] ? ~0 : 0,
                a.i[5] == b.i[5] ? ~0 : 0,
                a.i[6] == b.i[6] ? ~0 : 0,
                a.i[7] == b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i64> cmpNe(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0,
                a.i[4] != b.i[4] ? ~0 : 0,
                a.i[5] != b.i[5] ? ~0 : 0,
                a.i[6] != b.i[6] ? ~0 : 0,
                a.i[7] != b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i64> cmpLt(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0,
                a.i[4] < b.i[4] ? ~0 : 0,
                a.i[5] < b.i[5] ? ~0 : 0,
                a.i[6] < b.i[6] ? ~0 : 0,
                a.i[7] < b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i64> cmpLe(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0,
                a.i[4] <= b.i[4] ? ~0 : 0,
                a.i[5] <= b.i[5] ? ~0 : 0,
                a.i[6] <= b.i[6] ? ~0 : 0,
                a.i[7] <= b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i64> cmpGt(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0,
                a.i[4] > b.i[4] ? ~0 : 0,
                a.i[5] > b.i[5] ? ~0 : 0,
                a.i[6] > b.i[6] ? ~0 : 0,
                a.i[7] > b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i64> cmpGe(const Simd<8, i64> &a, const Simd<8, i64> &b) {
            return Simd<8, i64>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0,
                a.i[4] >= b.i[4] ? ~0 : 0,
                a.i[5] >= b.i[5] ? ~0 : 0,
                a.i[6] >= b.i[6] ? ~0 : 0,
                a.i[7] >= b.i[7] ? ~0 : 0);
        }

        static inline u64 movemask(const Simd<8, i64> &a) {
            u64 mask = 0;

            mask |= (((u64)a.i[0] >> 7) & 1) << 0;
            mask |= (((u64)a.i[0] >> 15) & 1) << 1;
            mask |= (((u64)a.i[0] >> 23) & 1) << 2;
            mask |= (((u64)a.i[0] >> 31) & 1) << 3;

            mask |= (((u64)a.i[1] >> 7) & 1) << 4;
            mask |= (((u64)a.i[1] >> 15) & 1) << 5;
            mask |= (((u64)a.i[1] >> 23) & 1) << 6;
            mask |= (((u64)a.i[1] >> 31) & 1) << 7;

            mask |= (((u64)a.i[2] >> 7) & 1) << 8;
            mask |= (((u64)a.i[2] >> 15) & 1) << 9;
            mask |= (((u64)a.i[2] >> 23) & 1) << 10;
            mask |= (((u64)a.i[2] >> 31) & 1) << 11;

            mask |= (((u64)a.i[3] >> 7) & 1) << 12;
            mask |= (((u64)a.i[3] >> 15) & 1) << 13;
            mask |= (((u64)a.i[3] >> 23) & 1) << 14;
            mask |= (((u64)a.i[3] >> 31) & 1) << 15;

            mask |= (((u64)a.i[4] >> 7) & 1) << 16;
            mask |= (((u64)a.i[4] >> 15) & 1) << 17;
            mask |= (((u64)a.i[4] >> 23) & 1) << 18;
            mask |= (((u64)a.i[4] >> 31) & 1) << 19;

            mask |= (((u64)a.i[5] >> 7) & 1) << 20;
            mask |= (((u64)a.i[5] >> 15) & 1) << 21;
            mask |= (((u64)a.i[5] >> 23) & 1) << 22;
            mask |= (((u64)a.i[5] >> 31) & 1) << 23;

            mask |= (((u64)a.i[6] >> 7) & 1) << 24;
            mask |= (((u64)a.i[6] >> 15) & 1) << 25;
            mask |= (((u64)a.i[6] >> 23) & 1) << 26;
            mask |= (((u64)a.i[6] >> 31) & 1) << 27;

            mask |= (((u64)a.i[7] >> 7) & 1) << 28;
            mask |= (((u64)a.i[7] >> 15) & 1) << 29;
            mask |= (((u64)a.i[7] >> 23) & 1) << 30;
            mask |= (((u64)a.i[7] >> 31) & 1) << 31;

            mask |= (((u64)a.i[8] >> 7) & 1) << 32;
            mask |= (((u64)a.i[8] >> 15) & 1) << 33;
            mask |= (((u64)a.i[8] >> 23) & 1) << 34;
            mask |= (((u64)a.i[8] >> 31) & 1) << 35;

            mask |= (((u64)a.i[9] >> 7) & 1) << 36;
            mask |= (((u64)a.i[9] >> 15) & 1) << 37;
            mask |= (((u64)a.i[9] >> 23) & 1) << 38;
            mask |= (((u64)a.i[9] >> 31) & 1) << 39;

            mask |= (((u64)a.i[10] >> 7) & 1) << 40;
            mask |= (((u64)a.i[10] >> 15) & 1) << 41;
            mask |= (((u64)a.i[10] >> 23) & 1) << 42;
            mask |= (((u64)a.i[10] >> 31) & 1) << 43;

            mask |= (((u64)a.i[11] >> 7) & 1) << 44;
            mask |= (((u64)a.i[11] >> 15) & 1) << 45;
            mask |= (((u64)a.i[11] >> 23) & 1) << 46;
            mask |= (((u64)a.i[11] >> 31) & 1) << 47;

            mask |= (((u64)a.i[12] >> 7) & 1) << 48;
            mask |= (((u64)a.i[12] >> 15) & 1) << 49;
            mask |= (((u64)a.i[12] >> 23) & 1) << 50;
            mask |= (((u64)a.i[12] >> 31) & 1) << 51;

            mask |= (((u64)a.i[13] >> 7) & 1) << 52;
            mask |= (((u64)a.i[13] >> 15) & 1) << 53;
            mask |= (((u64)a.i[13] >> 23) & 1) << 54;
            mask |= (((u64)a.i[13] >> 31) & 1) << 55;

            mask |= (((u64)a.i[14] >> 7) & 1) << 56;
            mask |= (((u64)a.i[14] >> 15) & 1) << 57;
            mask |= (((u64)a.i[14] >> 23) & 1) << 58;
            mask |= (((u64)a.i[14] >> 31) & 1) << 59;

            mask |= (((u64)a.i[15] >> 7) & 1) << 60;
            mask |= (((u64)a.i[15] >> 15) & 1) << 61;
            mask |= (((u64)a.i[15] >> 23) & 1) << 62;
            mask |= (((u64)a.i[15] >> 31) & 1) << 63;

            return mask;
        }
    };

}

#endif
