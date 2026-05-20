#ifndef AXN_PRTN_SIMD_GENERIC_I32_H
#define AXN_PRTN_SIMD_GENERIC_I32_H

/**
 * @file SIMD/Generic/i32.h
 * @brief Generic (scalar fallback) implementations for 32-bit integer SIMD.
 *
 * Provides `Proton::Simd<4,i32>` specializations and the usual integer
 * operations (load/store, add/sub, shifts, bitwise ops, comparisons).
 */

#include <math.h> // IWYU pragma: keep

#include "../Simd.h"
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<4, i32> {
        i32 i[4];
        Simd() {}
        Simd(i32 a0, i32 a1, i32 a2, i32 a3) {
            i[0] = a0;
            i[1] = a1;
            i[2] = a2;
            i[3] = a3;
        }

        static inline Simd<4, i32> load(const i32* ptr) {
            return Simd<4, i32>(ptr[0], ptr[1], ptr[2], ptr[3]);
        }
        static inline void store(i32* ptr, const Simd<4, i32> &simd) {
            ptr[0] = simd.i[0];
            ptr[1] = simd.i[1];
            ptr[2] = simd.i[2];
            ptr[3] = simd.i[3];
        }
        static inline Simd<4, i32> set1(const i32 value) {
            return Simd<4, i32>(value, value, value, value);
        }
        static inline Simd<4, i32> set(const i32 a0, const i32 a1, const i32 a2, const i32 a3) {
            return Simd<4, i32>(a0, a1, a2, a3);
        }

        static inline Simd<4, i32> add(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] + b.i[0], a.i[1] + b.i[1], a.i[2] + b.i[2], a.i[3] + b.i[3]);
        }
        static inline Simd<4, i32> sub(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] - b.i[0], a.i[1] - b.i[1], a.i[2] - b.i[2], a.i[3] - b.i[3]);
        }
        static inline Simd<4, i32> abs(const Simd<4, i32> &a) {
            return Simd<4, i32>(labs(a.i[0]), labs(a.i[1]), labs(a.i[2]), labs(a.i[3]));
        }
        static inline Simd<4, i32> neg(const Simd<4, i32> &a) {
            return Simd<4, i32>(-a.i[0], -a.i[1], -a.i[2], -a.i[3]);
        }
        static inline Simd<4, i32> min(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3]);
        }
        static inline Simd<4, i32> max(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3]);
        }

        static inline Simd<4, i32> land(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] & b.i[0], a.i[1] & b.i[1], a.i[2] & b.i[2], a.i[3] & b.i[3]);
        }
        static inline Simd<4, i32> lor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] | b.i[0], a.i[1] | b.i[1], a.i[2] | b.i[2], a.i[3] | b.i[3]);
        }
        static inline Simd<4, i32> lxor(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] ^ b.i[0], a.i[1] ^ b.i[1], a.i[2] ^ b.i[2], a.i[3] ^ b.i[3]);
        }
        static inline Simd<4, i32> lnot(const Simd<4, i32> &a) {
            return Simd<4, i32>(~a.i[0], ~a.i[1], ~a.i[2], ~a.i[3]);
        }
        static inline Simd<4, i32> landnot(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] & ~b.i[0], a.i[1] & ~b.i[1], a.i[2] & ~b.i[2], a.i[3] & ~b.i[3]);
        }

        static inline Simd<4, i32> shl(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(a.i[0] << count, a.i[1] << count, a.i[2] << count, a.i[3] << count);
        }
        static inline Simd<4, i32> shr(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(
                (i32)(((u32)a.i[0]) >> count),
                (i32)(((u32)a.i[1]) >> count),
                (i32)(((u32)a.i[2]) >> count),
                (i32)(((u32)a.i[3]) >> count));
        }
        static inline Simd<4, i32> sar(const Simd<4, i32> &a, i32 count) {
            return Simd<4, i32>(a.i[0] >> count, a.i[1] >> count, a.i[2] >> count, a.i[3] >> count);
        }

        static inline Simd<4, i32> mul(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(a.i[0] * b.i[0], a.i[1] * b.i[1], a.i[2] * b.i[2], a.i[3] * b.i[3]);
        }

        static inline Simd<4, i32> cmpEq(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i32> cmpNe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i32> cmpLt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i32> cmpLe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i32> cmpGt(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0);
        }
        static inline Simd<4, i32> cmpGe(const Simd<4, i32> &a, const Simd<4, i32> &b) {
            return Simd<4, i32>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0);
        }

        static inline u32 movemask(const Simd<4, i32> &a) {
            i32 mask = 0;

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
    union Simd<8, i32> {
        i32 i[8];
        Simd() {}
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
            return Simd<8, i32>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
        }
        static inline void store(i32* ptr, const Simd<8, i32> &simd) {
            ptr[0] = simd.i[0];
            ptr[1] = simd.i[1];
            ptr[2] = simd.i[2];
            ptr[3] = simd.i[3];
            ptr[4] = simd.i[4];
            ptr[5] = simd.i[5];
            ptr[6] = simd.i[6];
            ptr[7] = simd.i[7];
        }
        static inline Simd<8, i32> set1(const i32 value) {
            return Simd<8, i32>(value, value, value, value, value, value, value, value);
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
            return Simd<8, i32>(a0, a1, a2, a3, a4, a5, a6, a7);
        }

        static inline Simd<8, i32> add(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] + b.i[0],
                a.i[1] + b.i[1],
                a.i[2] + b.i[2],
                a.i[3] + b.i[3],
                a.i[4] + b.i[4],
                a.i[5] + b.i[5],
                a.i[6] + b.i[6],
                a.i[7] + b.i[7]);
        }
        static inline Simd<8, i32> sub(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] - b.i[0],
                a.i[1] - b.i[1],
                a.i[2] - b.i[2],
                a.i[3] - b.i[3],
                a.i[4] - b.i[4],
                a.i[5] - b.i[5],
                a.i[6] - b.i[6],
                a.i[7] - b.i[7]);
        }
        static inline Simd<8, i32> abs(const Simd<8, i32> &a) {
            return Simd<8, i32>(
                labs(a.i[0]),
                labs(a.i[1]),
                labs(a.i[2]),
                labs(a.i[3]),
                labs(a.i[4]),
                labs(a.i[5]),
                labs(a.i[6]),
                labs(a.i[7]));
        }
        static inline Simd<8, i32> neg(const Simd<8, i32> &a) {
            return Simd<8, i32>(
                -a.i[0], -a.i[1], -a.i[2], -a.i[3], -a.i[4], -a.i[5], -a.i[6], -a.i[7]);
        }
        static inline Simd<8, i32> min(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] < b.i[0] ? a.i[0] : b.i[0],
                a.i[1] < b.i[1] ? a.i[1] : b.i[1],
                a.i[2] < b.i[2] ? a.i[2] : b.i[2],
                a.i[3] < b.i[3] ? a.i[3] : b.i[3],
                a.i[4] < b.i[4] ? a.i[4] : b.i[4],
                a.i[5] < b.i[5] ? a.i[5] : b.i[5],
                a.i[6] < b.i[6] ? a.i[6] : b.i[6],
                a.i[7] < b.i[7] ? a.i[7] : b.i[7]);
        }
        static inline Simd<8, i32> max(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] > b.i[0] ? a.i[0] : b.i[0],
                a.i[1] > b.i[1] ? a.i[1] : b.i[1],
                a.i[2] > b.i[2] ? a.i[2] : b.i[2],
                a.i[3] > b.i[3] ? a.i[3] : b.i[3],
                a.i[4] > b.i[4] ? a.i[4] : b.i[4],
                a.i[5] > b.i[5] ? a.i[5] : b.i[5],
                a.i[6] > b.i[6] ? a.i[6] : b.i[6],
                a.i[7] > b.i[7] ? a.i[7] : b.i[7]);
        }

        static inline Simd<8, i32> land(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] & b.i[0],
                a.i[1] & b.i[1],
                a.i[2] & b.i[2],
                a.i[3] & b.i[3],
                a.i[4] & b.i[4],
                a.i[5] & b.i[5],
                a.i[6] & b.i[6],
                a.i[7] & b.i[7]);
        }
        static inline Simd<8, i32> lor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] | b.i[0],
                a.i[1] | b.i[1],
                a.i[2] | b.i[2],
                a.i[3] | b.i[3],
                a.i[4] | b.i[4],
                a.i[5] | b.i[5],
                a.i[6] | b.i[6],
                a.i[7] | b.i[7]);
        }
        static inline Simd<8, i32> lxor(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] ^ b.i[0],
                a.i[1] ^ b.i[1],
                a.i[2] ^ b.i[2],
                a.i[3] ^ b.i[3],
                a.i[4] ^ b.i[4],
                a.i[5] ^ b.i[5],
                a.i[6] ^ b.i[6],
                a.i[7] ^ b.i[7]);
        }
        static inline Simd<8, i32> lnot(const Simd<8, i32> &a) {
            return Simd<8, i32>(
                ~a.i[0], ~a.i[1], ~a.i[2], ~a.i[3], ~a.i[4], ~a.i[5], ~a.i[6], ~a.i[7]);
        }
        static inline Simd<8, i32> landnot(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] & ~b.i[0],
                a.i[1] & ~b.i[1],
                a.i[2] & ~b.i[2],
                a.i[3] & ~b.i[3],
                a.i[4] & ~b.i[4],
                a.i[5] & ~b.i[5],
                a.i[6] & ~b.i[6],
                a.i[7] & ~b.i[7]);
        }

        static inline Simd<8, i32> shl(const Simd<8, i32> &a, i32 count) {
            return Simd<8, i32>(
                a.i[0] << count,
                a.i[1] << count,
                a.i[2] << count,
                a.i[3] << count,
                a.i[4] << count,
                a.i[5] << count,
                a.i[6] << count,
                a.i[7] << count);
        }
        static inline Simd<8, i32> shr(const Simd<8, i32> &a, i32 count) {
            return Simd<8, i32>(
                (i32)(((u32)a.i[0]) >> count),
                (i32)(((u32)a.i[1]) >> count),
                (i32)(((u32)a.i[2]) >> count),
                (i32)(((u32)a.i[3]) >> count),
                (i32)(((u32)a.i[4]) >> count),
                (i32)(((u32)a.i[5]) >> count),
                (i32)(((u32)a.i[6]) >> count),
                (i32)(((u32)a.i[7]) >> count));
        }
        static inline Simd<8, i32> sar(const Simd<8, i32> &a, i32 count) {
            return Simd<8, i32>(
                a.i[0] >> count,
                a.i[1] >> count,
                a.i[2] >> count,
                a.i[3] >> count,
                a.i[4] >> count,
                a.i[5] >> count,
                a.i[6] >> count,
                a.i[7] >> count);
        }

        static inline Simd<8, i32> mul(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] * b.i[0],
                a.i[1] * b.i[1],
                a.i[2] * b.i[2],
                a.i[3] * b.i[3],
                a.i[4] * b.i[4],
                a.i[5] * b.i[5],
                a.i[6] * b.i[6],
                a.i[7] * b.i[7]);
        }

        static inline Simd<8, i32> cmpEq(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] == b.i[0] ? ~0 : 0,
                a.i[1] == b.i[1] ? ~0 : 0,
                a.i[2] == b.i[2] ? ~0 : 0,
                a.i[3] == b.i[3] ? ~0 : 0,
                a.i[4] == b.i[4] ? ~0 : 0,
                a.i[5] == b.i[5] ? ~0 : 0,
                a.i[6] == b.i[6] ? ~0 : 0,
                a.i[7] == b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i32> cmpNe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] != b.i[0] ? ~0 : 0,
                a.i[1] != b.i[1] ? ~0 : 0,
                a.i[2] != b.i[2] ? ~0 : 0,
                a.i[3] != b.i[3] ? ~0 : 0,
                a.i[4] != b.i[4] ? ~0 : 0,
                a.i[5] != b.i[5] ? ~0 : 0,
                a.i[6] != b.i[6] ? ~0 : 0,
                a.i[7] != b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i32> cmpLt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] < b.i[0] ? ~0 : 0,
                a.i[1] < b.i[1] ? ~0 : 0,
                a.i[2] < b.i[2] ? ~0 : 0,
                a.i[3] < b.i[3] ? ~0 : 0,
                a.i[4] < b.i[4] ? ~0 : 0,
                a.i[5] < b.i[5] ? ~0 : 0,
                a.i[6] < b.i[6] ? ~0 : 0,
                a.i[7] < b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i32> cmpLe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] <= b.i[0] ? ~0 : 0,
                a.i[1] <= b.i[1] ? ~0 : 0,
                a.i[2] <= b.i[2] ? ~0 : 0,
                a.i[3] <= b.i[3] ? ~0 : 0,
                a.i[4] <= b.i[4] ? ~0 : 0,
                a.i[5] <= b.i[5] ? ~0 : 0,
                a.i[6] <= b.i[6] ? ~0 : 0,
                a.i[7] <= b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i32> cmpGt(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] > b.i[0] ? ~0 : 0,
                a.i[1] > b.i[1] ? ~0 : 0,
                a.i[2] > b.i[2] ? ~0 : 0,
                a.i[3] > b.i[3] ? ~0 : 0,
                a.i[4] > b.i[4] ? ~0 : 0,
                a.i[5] > b.i[5] ? ~0 : 0,
                a.i[6] > b.i[6] ? ~0 : 0,
                a.i[7] > b.i[7] ? ~0 : 0);
        }
        static inline Simd<8, i32> cmpGe(const Simd<8, i32> &a, const Simd<8, i32> &b) {
            return Simd<8, i32>(
                a.i[0] >= b.i[0] ? ~0 : 0,
                a.i[1] >= b.i[1] ? ~0 : 0,
                a.i[2] >= b.i[2] ? ~0 : 0,
                a.i[3] >= b.i[3] ? ~0 : 0,
                a.i[4] >= b.i[4] ? ~0 : 0,
                a.i[5] >= b.i[5] ? ~0 : 0,
                a.i[6] >= b.i[6] ? ~0 : 0,
                a.i[7] >= b.i[7] ? ~0 : 0);
        }

        static inline u32 movemask(const Simd<8, i32> &a) {
            i32 mask = 0;

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
    union Simd<16, i32> {
        i32 i[16];
        Simd() {}
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
            return Simd<16, i32>(
                ptr[0],
                ptr[1],
                ptr[2],
                ptr[3],
                ptr[4],
                ptr[5],
                ptr[6],
                ptr[7],
                ptr[8],
                ptr[9],
                ptr[10],
                ptr[11],
                ptr[12],
                ptr[13],
                ptr[14],
                ptr[15]);
        }
        static inline void store(i32* ptr, const Simd<16, i32> &simd) {
            ptr[0]  = simd.i[0];
            ptr[1]  = simd.i[1];
            ptr[2]  = simd.i[2];
            ptr[3]  = simd.i[3];
            ptr[4]  = simd.i[4];
            ptr[5]  = simd.i[5];
            ptr[6]  = simd.i[6];
            ptr[7]  = simd.i[7];
            ptr[8]  = simd.i[8];
            ptr[9]  = simd.i[9];
            ptr[10] = simd.i[10];
            ptr[11] = simd.i[11];
            ptr[12] = simd.i[12];
            ptr[13] = simd.i[13];
            ptr[14] = simd.i[14];
            ptr[15] = simd.i[15];
        }
        static inline Simd<16, i32> set1(const i32 value) {
            return Simd<16, i32>(
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value,
                value);
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
            return Simd<16, i32>(
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
        }

        static inline Simd<16, i32> add(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> sub(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> abs(const Simd<16, i32> &a) {
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
        }
        static inline Simd<16, i32> neg(const Simd<16, i32> &a) {
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
        }
        static inline Simd<16, i32> min(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> max(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }

        static inline Simd<16, i32> land(const Simd<16, i32> &a, const Simd<16, i32> &b) {
            return Simd<16, i32>(
                a.i[0] & b.i[0],
                a.i[1] & b.i[1],
                a.i[2] & b.i[2],
                a.i[3] & b.i[3],
                a.i[4] & b.i[4],
                a.i[5] & b.i[5],
                a.i[6] & b.i[6],
                a.i[7] & b.i[7],
                a.i[8] & b.i[8],
                a.i[9] & b.i[9],
                a.i[10] & b.i[10],
                a.i[11] & b.i[11],
                a.i[12] & b.i[12],
                a.i[13] & b.i[13],
                a.i[14] & b.i[14],
                a.i[15] & b.i[15]);
        }
        static inline Simd<16, i32> lor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
            return Simd<16, i32>(
                a.i[0] | b.i[0],
                a.i[1] | b.i[1],
                a.i[2] | b.i[2],
                a.i[3] | b.i[3],
                a.i[4] | b.i[4],
                a.i[5] | b.i[5],
                a.i[6] | b.i[6],
                a.i[7] | b.i[7],
                a.i[8] | b.i[8],
                a.i[9] | b.i[9],
                a.i[10] | b.i[10],
                a.i[11] | b.i[11],
                a.i[12] | b.i[12],
                a.i[13] | b.i[13],
                a.i[14] | b.i[14],
                a.i[15] | b.i[15]);
        }
        static inline Simd<16, i32> lxor(const Simd<16, i32> &a, const Simd<16, i32> &b) {
            return Simd<16, i32>(
                a.i[0] ^ b.i[0],
                a.i[1] ^ b.i[1],
                a.i[2] ^ b.i[2],
                a.i[3] ^ b.i[3],
                a.i[4] ^ b.i[4],
                a.i[5] ^ b.i[5],
                a.i[6] ^ b.i[6],
                a.i[7] ^ b.i[7],
                a.i[8] ^ b.i[8],
                a.i[9] ^ b.i[9],
                a.i[10] ^ b.i[10],
                a.i[11] ^ b.i[11],
                a.i[12] ^ b.i[12],
                a.i[13] ^ b.i[13],
                a.i[14] ^ b.i[14],
                a.i[15] ^ b.i[15]);
        }
        static inline Simd<16, i32> lnot(const Simd<16, i32> &a) {
            return Simd<16, i32>(
                ~a.i[0],
                ~a.i[1],
                ~a.i[2],
                ~a.i[3],
                ~a.i[4],
                ~a.i[5],
                ~a.i[6],
                ~a.i[7],
                ~a.i[8],
                ~a.i[9],
                ~a.i[10],
                ~a.i[11],
                ~a.i[12],
                ~a.i[13],
                ~a.i[14],
                ~a.i[15]);
        }
        static inline Simd<16, i32> landnot(const Simd<16, i32> &a, const Simd<16, i32> &b) {
            return Simd<16, i32>(
                a.i[0] & ~b.i[0],
                a.i[1] & ~b.i[1],
                a.i[2] & ~b.i[2],
                a.i[3] & ~b.i[3],
                a.i[4] & ~b.i[4],
                a.i[5] & ~b.i[5],
                a.i[6] & ~b.i[6],
                a.i[7] & ~b.i[7],
                a.i[8] & ~b.i[8],
                a.i[9] & ~b.i[9],
                a.i[10] & ~b.i[10],
                a.i[11] & ~b.i[11],
                a.i[12] & ~b.i[12],
                a.i[13] & ~b.i[13],
                a.i[14] & ~b.i[14],
                a.i[15] & ~b.i[15]);
        }

        static inline Simd<16, i32> shl(const Simd<16, i32> &a, i32 count) {
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
        }
        static inline Simd<16, i32> shr(const Simd<16, i32> &a, i32 count) {
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
        }
        static inline Simd<16, i32> sar(const Simd<16, i32> &a, i32 count) {
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
        }

        static inline Simd<16, i32> mul(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }

        static inline Simd<16, i32> cmpEq(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> cmpNe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> cmpLt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> cmpLe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> cmpGt(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }
        static inline Simd<16, i32> cmpGe(const Simd<16, i32> &a, const Simd<16, i32> &b) {
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
        }

        static inline u64 movemask(const Simd<16, i32> &a) {
            u64 mask = 0;

            mask |= ((u64)(a.i[0] >> 7) & 1) << 0;
            mask |= ((u64)(a.i[0] >> 15) & 1) << 1;
            mask |= ((u64)(a.i[0] >> 23) & 1) << 2;
            mask |= ((u64)(a.i[0] >> 31) & 1) << 3;

            mask |= ((u64)(a.i[1] >> 7) & 1) << 4;
            mask |= ((u64)(a.i[1] >> 15) & 1) << 5;
            mask |= ((u64)(a.i[1] >> 23) & 1) << 6;
            mask |= ((u64)(a.i[1] >> 31) & 1) << 7;

            mask |= ((u64)(a.i[2] >> 7) & 1) << 8;
            mask |= ((u64)(a.i[2] >> 15) & 1) << 9;
            mask |= ((u64)(a.i[2] >> 23) & 1) << 10;
            mask |= ((u64)(a.i[2] >> 31) & 1) << 11;

            mask |= ((u64)(a.i[3] >> 7) & 1) << 12;
            mask |= ((u64)(a.i[3] >> 15) & 1) << 13;
            mask |= ((u64)(a.i[3] >> 23) & 1) << 14;
            mask |= ((u64)(a.i[3] >> 31) & 1) << 15;

            mask |= ((u64)(a.i[4] >> 7) & 1) << 16;
            mask |= ((u64)(a.i[4] >> 15) & 1) << 17;
            mask |= ((u64)(a.i[4] >> 23) & 1) << 18;
            mask |= ((u64)(a.i[4] >> 31) & 1) << 19;

            mask |= ((u64)(a.i[5] >> 7) & 1) << 20;
            mask |= ((u64)(a.i[5] >> 15) & 1) << 21;
            mask |= ((u64)(a.i[5] >> 23) & 1) << 22;
            mask |= ((u64)(a.i[5] >> 31) & 1) << 23;
            mask |= ((u64)(a.i[6] >> 7) & 1) << 24;
            mask |= ((u64)(a.i[6] >> 15) & 1) << 25;
            mask |= ((u64)(a.i[6] >> 23) & 1) << 26;
            mask |= ((u64)(a.i[6] >> 31) & 1) << 27;
            mask |= ((u64)(a.i[7] >> 7) & 1) << 28;
            mask |= ((u64)(a.i[7] >> 15) & 1) << 29;
            mask |= ((u64)(a.i[7] >> 23) & 1) << 30;
            mask |= ((u64)(a.i[7] >> 31) & 1) << 31;
            mask |= ((u64)(a.i[8] >> 7) & 1) << 32;
            mask |= ((u64)(a.i[8] >> 15) & 1) << 33;
            mask |= ((u64)(a.i[8] >> 23) & 1) << 34;
            mask |= ((u64)(a.i[8] >> 31) & 1) << 35;
            mask |= ((u64)(a.i[9] >> 7) & 1) << 36;
            mask |= ((u64)(a.i[9] >> 15) & 1) << 37;
            mask |= ((u64)(a.i[9] >> 23) & 1) << 38;
            mask |= ((u64)(a.i[9] >> 31) & 1) << 39;
            mask |= ((u64)(a.i[10] >> 7) & 1) << 40;
            mask |= ((u64)(a.i[10] >> 15) & 1) << 41;
            mask |= ((u64)(a.i[10] >> 23) & 1) << 42;
            mask |= ((u64)(a.i[10] >> 31) & 1) << 43;
            mask |= ((u64)(a.i[11] >> 7) & 1) << 44;
            mask |= ((u64)(a.i[11] >> 15) & 1) << 45;
            mask |= ((u64)(a.i[11] >> 23) & 1) << 46;
            mask |= ((u64)(a.i[11] >> 31) & 1) << 47;
            mask |= ((u64)(a.i[12] >> 7) & 1) << 48;
            mask |= ((u64)(a.i[12] >> 15) & 1) << 49;
            mask |= ((u64)(a.i[12] >> 23) & 1) << 50;
            mask |= ((u64)(a.i[12] >> 31) & 1) << 51;
            mask |= ((u64)(a.i[13] >> 7) & 1) << 52;
            mask |= ((u64)(a.i[13] >> 15) & 1) << 53;
            mask |= ((u64)(a.i[13] >> 23) & 1) << 54;
            mask |= ((u64)(a.i[13] >> 31) & 1) << 55;
            mask |= ((u64)(a.i[14] >> 7) & 1) << 56;
            mask |= ((u64)(a.i[14] >> 15) & 1) << 57;
            mask |= ((u64)(a.i[14] >> 23) & 1) << 58;
            mask |= ((u64)(a.i[14] >> 31) & 1) << 59;
            mask |= ((u64)(a.i[15] >> 7) & 1) << 60;
            mask |= ((u64)(a.i[15] >> 15) & 1) << 61;
            mask |= ((u64)(a.i[15] >> 23) & 1) << 62;
            mask |= ((u64)(a.i[15] >> 31) & 1) << 63;

            return mask;
        }
    };

}

#endif
