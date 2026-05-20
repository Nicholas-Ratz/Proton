#ifndef AXN_PRTN_SIMD_GENERIC_F32_H
#define AXN_PRTN_SIMD_GENERIC_F32_H

/**
 * @file SIMD/Generic/f32.h
 * @brief Generic (scalar fallback) implementations for 32-bit float SIMD.
 *
 * Provides `Proton::Simd<4,f32>` and `Proton::Simd<8,f32>` specializations
 * along with common operations: load/store, set/set1, arithmetic, bitwise,
 * comparisons and utility functions (sqrt, reciprocal, movemask).
 */

#include <math.h>

#include "../Simd.h"
#include <Quark/Types.h>

namespace Proton {
    template <>
    union Simd<4, f32> {
        f32 f[4];
        Simd() {}
        Simd(f32 a0, f32 a1, f32 a2, f32 a3) {
            f[0] = a0;
            f[1] = a1;
            f[2] = a2;
            f[3] = a3;
        }

        static inline Simd<4, f32> load(const f32* ptr) {
            return Simd<4, f32>(ptr[0], ptr[1], ptr[2], ptr[3]);
        }
        static inline void store(f32* ptr, const Simd<4, f32> &simd) {
            ptr[0] = simd.f[0];
            ptr[1] = simd.f[1];
            ptr[2] = simd.f[2];
            ptr[3] = simd.f[3];
        }
        static inline Simd<4, f32> set1(const f32 value) {
            return Simd<4, f32>(value, value, value, value);
        }
        static inline Simd<4, f32> set(const f32 a0, const f32 a1, const f32 a2, const f32 a3) {
            return Simd<4, f32>(a0, a1, a2, a3);
        }

        static inline Simd<4, f32> add(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(a.f[0] + b.f[0], a.f[1] + b.f[1], a.f[2] + b.f[2], a.f[3] + b.f[3]);
        }
        static inline Simd<4, f32> sub(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(a.f[0] - b.f[0], a.f[1] - b.f[1], a.f[2] - b.f[2], a.f[3] - b.f[3]);
        }
        static inline Simd<4, f32> neg(const Simd<4, f32> &a) {
            return Simd<4, f32>(-a.f[0], -a.f[1], -a.f[2], -a.f[3]);
        }
        static inline Simd<4, f32> abs(const Simd<4, f32> &a) {
            return Simd<4, f32>(fabsf(a.f[0]), fabsf(a.f[1]), fabsf(a.f[2]), fabsf(a.f[3]));
        }
        static inline Simd<4, f32> min(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3]);
        }
        static inline Simd<4, f32> max(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3]);
        }

        static inline Simd<4, f32> land(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];

            i32 result0 = ai0 & bi0;
            i32 result1 = ai1 & bi1;
            i32 result2 = ai2 & bi2;
            i32 result3 = ai3 & bi3;

            return Simd<4, f32>(*(f32*)&result0, *(f32*)&result1, *(f32*)&result2, *(f32*)&result3);
        }
        static inline Simd<4, f32> lor(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];

            i32 result0 = ai0 | bi0;
            i32 result1 = ai1 | bi1;
            i32 result2 = ai2 | bi2;
            i32 result3 = ai3 | bi3;

            return Simd<4, f32>(*(f32*)&result0, *(f32*)&result1, *(f32*)&result2, *(f32*)&result3);
        }
        static inline Simd<4, f32> lxor(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];

            i32 result0 = ai0 ^ bi0;
            i32 result1 = ai1 ^ bi1;
            i32 result2 = ai2 ^ bi2;
            i32 result3 = ai3 ^ bi3;

            return Simd<4, f32>(*(f32*)&result0, *(f32*)&result1, *(f32*)&result2, *(f32*)&result3);
        }
        static inline Simd<4, f32> lnot(const Simd<4, f32> &a) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];

            i32 result0 = ~ai0;
            i32 result1 = ~ai1;
            i32 result2 = ~ai2;
            i32 result3 = ~ai3;

            return Simd<4, f32>(*(f32*)&result0, *(f32*)&result1, *(f32*)&result2, *(f32*)&result3);
        }
        static inline Simd<4, f32> landnot(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];

            i32 result0 = ai0 & ~bi0;
            i32 result1 = ai1 & ~bi1;
            i32 result2 = ai2 & ~bi2;
            i32 result3 = ai3 & ~bi3;

            return Simd<4, f32>(*(f32*)&result0, *(f32*)&result1, *(f32*)&result2, *(f32*)&result3);
        }

        static inline Simd<4, f32> mul(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(a.f[0] * b.f[0], a.f[1] * b.f[1], a.f[2] * b.f[2], a.f[3] * b.f[3]);
        }
        static inline Simd<4, f32> div(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            return Simd<4, f32>(a.f[0] / b.f[0], a.f[1] / b.f[1], a.f[2] / b.f[2], a.f[3] / b.f[3]);
        }
        static inline Simd<4, f32> sqrt(const Simd<4, f32> &a) {
            return Simd<4, f32>(sqrtf(a.f[0]), sqrtf(a.f[1]), sqrtf(a.f[2]), sqrtf(a.f[3]));
        }
        static inline Simd<4, f32> rsqrt(const Simd<4, f32> &a) {
            return Simd<4, f32>(
                1.0f / sqrtf(a.f[0]),
                1.0f / sqrtf(a.f[1]),
                1.0f / sqrtf(a.f[2]),
                1.0f / sqrtf(a.f[3]));
        }
        static inline Simd<4, f32> reciprocal(const Simd<4, f32> &a) {
            return Simd<4, f32>(1.0f / a.f[0], 1.0f / a.f[1], 1.0f / a.f[2], 1.0f / a.f[3]);
        }

        static inline Simd<4, f32> cmpEq(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] == b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] == b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] == b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] == b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }
        static inline Simd<4, f32> cmpNe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] != b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] != b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] != b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] != b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }
        static inline Simd<4, f32> cmpLt(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] < b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] < b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] < b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] < b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }
        static inline Simd<4, f32> cmpLe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] <= b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] <= b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] <= b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] <= b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }
        static inline Simd<4, f32> cmpGt(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] > b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] > b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] > b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] > b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }
        static inline Simd<4, f32> cmpGe(const Simd<4, f32> &a, const Simd<4, f32> &b) {
            i32 r0 = a.f[0] >= b.f[0] ? ~0 : 0;
            i32 r1 = a.f[1] >= b.f[1] ? ~0 : 0;
            i32 r2 = a.f[2] >= b.f[2] ? ~0 : 0;
            i32 r3 = a.f[3] >= b.f[3] ? ~0 : 0;
            return Simd<4, f32>(*(f32*)&r0, *(f32*)&r1, *(f32*)&r2, *(f32*)&r3);
        }

        static inline u32 movemask(const Simd<4, f32> &a) {
            u32 mask = 0;

            mask |= ((*(u32*)&a.f[0] >> 7) & 1) << 0;
            mask |= ((*(u32*)&a.f[0] >> 15) & 1) << 1;
            mask |= ((*(u32*)&a.f[0] >> 23) & 1) << 2;
            mask |= ((*(u32*)&a.f[0] >> 31) & 1) << 3;

            mask |= ((*(u32*)&a.f[1] >> 7) & 1) << 4;
            mask |= ((*(u32*)&a.f[1] >> 15) & 1) << 5;
            mask |= ((*(u32*)&a.f[1] >> 23) & 1) << 6;
            mask |= ((*(u32*)&a.f[1] >> 31) & 1) << 7;

            mask |= ((*(u32*)&a.f[2] >> 7) & 1) << 8;
            mask |= ((*(u32*)&a.f[2] >> 15) & 1) << 9;
            mask |= ((*(u32*)&a.f[2] >> 23) & 1) << 10;
            mask |= ((*(u32*)&a.f[2] >> 31) & 1) << 11;

            mask |= ((*(u32*)&a.f[3] >> 7) & 1) << 12;
            mask |= ((*(u32*)&a.f[3] >> 15) & 1) << 13;
            mask |= ((*(u32*)&a.f[3] >> 23) & 1) << 14;
            mask |= ((*(u32*)&a.f[3] >> 31) & 1) << 15;

            return mask;
        }
    };

    template <>
    union Simd<8, f32> {
        f32 f[8];
        Simd() {}
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
            return Simd<8, f32>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
        }
        static inline void store(f32* ptr, const Simd<8, f32> &simd) {
            ptr[0] = simd.f[0];
            ptr[1] = simd.f[1];
            ptr[2] = simd.f[2];
            ptr[3] = simd.f[3];
            ptr[4] = simd.f[4];
            ptr[5] = simd.f[5];
            ptr[6] = simd.f[6];
            ptr[7] = simd.f[7];
        }
        static inline Simd<8, f32> set1(const f32 value) {
            return Simd<8, f32>(value, value, value, value, value, value, value, value);
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
            return Simd<8, f32>(a0, a1, a2, a3, a4, a5, a6, a7);
        }

        static inline Simd<8, f32> add(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] + b.f[0],
                a.f[1] + b.f[1],
                a.f[2] + b.f[2],
                a.f[3] + b.f[3],
                a.f[4] + b.f[4],
                a.f[5] + b.f[5],
                a.f[6] + b.f[6],
                a.f[7] + b.f[7]);
        }
        static inline Simd<8, f32> sub(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] - b.f[0],
                a.f[1] - b.f[1],
                a.f[2] - b.f[2],
                a.f[3] - b.f[3],
                a.f[4] - b.f[4],
                a.f[5] - b.f[5],
                a.f[6] - b.f[6],
                a.f[7] - b.f[7]);
        }
        static inline Simd<8, f32> neg(const Simd<8, f32> &a) {
            return Simd<8, f32>(
                -a.f[0], -a.f[1], -a.f[2], -a.f[3], -a.f[4], -a.f[5], -a.f[6], -a.f[7]);
        }
        static inline Simd<8, f32> abs(const Simd<8, f32> &a) {
            return Simd<8, f32>(
                fabsf(a.f[0]),
                fabsf(a.f[1]),
                fabsf(a.f[2]),
                fabsf(a.f[3]),
                fabsf(a.f[4]),
                fabsf(a.f[5]),
                fabsf(a.f[6]),
                fabsf(a.f[7]));
        }
        static inline Simd<8, f32> min(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3],
                a.f[4] < b.f[4] ? a.f[4] : b.f[4],
                a.f[5] < b.f[5] ? a.f[5] : b.f[5],
                a.f[6] < b.f[6] ? a.f[6] : b.f[6],
                a.f[7] < b.f[7] ? a.f[7] : b.f[7]);
        }
        static inline Simd<8, f32> max(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3],
                a.f[4] > b.f[4] ? a.f[4] : b.f[4],
                a.f[5] > b.f[5] ? a.f[5] : b.f[5],
                a.f[6] > b.f[6] ? a.f[6] : b.f[6],
                a.f[7] > b.f[7] ? a.f[7] : b.f[7]);
        }

        static inline Simd<8, f32> land(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 ai4 = *(i32*)&a.f[4];
            i32 ai5 = *(i32*)&a.f[5];
            i32 ai6 = *(i32*)&a.f[6];
            i32 ai7 = *(i32*)&a.f[7];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];
            i32 bi4 = *(i32*)&b.f[4];
            i32 bi5 = *(i32*)&b.f[5];
            i32 bi6 = *(i32*)&b.f[6];
            i32 bi7 = *(i32*)&b.f[7];

            i32 result0 = ai0 & bi0;
            i32 result1 = ai1 & bi1;
            i32 result2 = ai2 & bi2;
            i32 result3 = ai3 & bi3;
            i32 result4 = ai4 & bi4;
            i32 result5 = ai5 & bi5;
            i32 result6 = ai6 & bi6;
            i32 result7 = ai7 & bi7;

            return Simd<8, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7);
        }
        static inline Simd<8, f32> lor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 ai4 = *(i32*)&a.f[4];
            i32 ai5 = *(i32*)&a.f[5];
            i32 ai6 = *(i32*)&a.f[6];
            i32 ai7 = *(i32*)&a.f[7];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];
            i32 bi4 = *(i32*)&b.f[4];
            i32 bi5 = *(i32*)&b.f[5];
            i32 bi6 = *(i32*)&b.f[6];
            i32 bi7 = *(i32*)&b.f[7];

            i32 result0 = ai0 | bi0;
            i32 result1 = ai1 | bi1;
            i32 result2 = ai2 | bi2;
            i32 result3 = ai3 | bi3;
            i32 result4 = ai4 | bi4;
            i32 result5 = ai5 | bi5;
            i32 result6 = ai6 | bi6;
            i32 result7 = ai7 | bi7;

            return Simd<8, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7);
        }
        static inline Simd<8, f32> lxor(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 ai4 = *(i32*)&a.f[4];
            i32 ai5 = *(i32*)&a.f[5];
            i32 ai6 = *(i32*)&a.f[6];
            i32 ai7 = *(i32*)&a.f[7];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];
            i32 bi4 = *(i32*)&b.f[4];
            i32 bi5 = *(i32*)&b.f[5];
            i32 bi6 = *(i32*)&b.f[6];
            i32 bi7 = *(i32*)&b.f[7];

            i32 result0 = ai0 ^ bi0;
            i32 result1 = ai1 ^ bi1;
            i32 result2 = ai2 ^ bi2;
            i32 result3 = ai3 ^ bi3;
            i32 result4 = ai4 ^ bi4;
            i32 result5 = ai5 ^ bi5;
            i32 result6 = ai6 ^ bi6;
            i32 result7 = ai7 ^ bi7;

            return Simd<8, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7);
        }
        static inline Simd<8, f32> lnot(const Simd<8, f32> &a) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 ai4 = *(i32*)&a.f[4];
            i32 ai5 = *(i32*)&a.f[5];
            i32 ai6 = *(i32*)&a.f[6];
            i32 ai7 = *(i32*)&a.f[7];

            i32 result0 = ~ai0;
            i32 result1 = ~ai1;
            i32 result2 = ~ai2;
            i32 result3 = ~ai3;
            i32 result4 = ~ai4;
            i32 result5 = ~ai5;
            i32 result6 = ~ai6;
            i32 result7 = ~ai7;

            return Simd<8, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7);
        }
        static inline Simd<8, f32> landnot(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 ai0 = *(i32*)&a.f[0];
            i32 ai1 = *(i32*)&a.f[1];
            i32 ai2 = *(i32*)&a.f[2];
            i32 ai3 = *(i32*)&a.f[3];
            i32 ai4 = *(i32*)&a.f[4];
            i32 ai5 = *(i32*)&a.f[5];
            i32 ai6 = *(i32*)&a.f[6];
            i32 ai7 = *(i32*)&a.f[7];
            i32 bi0 = *(i32*)&b.f[0];
            i32 bi1 = *(i32*)&b.f[1];
            i32 bi2 = *(i32*)&b.f[2];
            i32 bi3 = *(i32*)&b.f[3];
            i32 bi4 = *(i32*)&b.f[4];
            i32 bi5 = *(i32*)&b.f[5];
            i32 bi6 = *(i32*)&b.f[6];
            i32 bi7 = *(i32*)&b.f[7];

            i32 result0 = ai0 & ~bi0;
            i32 result1 = ai1 & ~bi1;
            i32 result2 = ai2 & ~bi2;
            i32 result3 = ai3 & ~bi3;
            i32 result4 = ai4 & ~bi4;
            i32 result5 = ai5 & ~bi5;
            i32 result6 = ai6 & ~bi6;
            i32 result7 = ai7 & ~bi7;

            return Simd<8, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7);
        }

        static inline Simd<8, f32> mul(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] * b.f[0],
                a.f[1] * b.f[1],
                a.f[2] * b.f[2],
                a.f[3] * b.f[3],
                a.f[4] * b.f[4],
                a.f[5] * b.f[5],
                a.f[6] * b.f[6],
                a.f[7] * b.f[7]);
        }
        static inline Simd<8, f32> div(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            return Simd<8, f32>(
                a.f[0] / b.f[0],
                a.f[1] / b.f[1],
                a.f[2] / b.f[2],
                a.f[3] / b.f[3],
                a.f[4] / b.f[4],
                a.f[5] / b.f[5],
                a.f[6] / b.f[6],
                a.f[7] / b.f[7]);
        }
        static inline Simd<8, f32> sqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(
                sqrtf(a.f[0]),
                sqrtf(a.f[1]),
                sqrtf(a.f[2]),
                sqrtf(a.f[3]),
                sqrtf(a.f[4]),
                sqrtf(a.f[5]),
                sqrtf(a.f[6]),
                sqrtf(a.f[7]));
        }
        static inline Simd<8, f32> rsqrt(const Simd<8, f32> &a) {
            return Simd<8, f32>(
                1.0f / sqrtf(a.f[0]),
                1.0f / sqrtf(a.f[1]),
                1.0f / sqrtf(a.f[2]),
                1.0f / sqrtf(a.f[3]),
                1.0f / sqrtf(a.f[4]),
                1.0f / sqrtf(a.f[5]),
                1.0f / sqrtf(a.f[6]),
                1.0f / sqrtf(a.f[7]));
        }
        static inline Simd<8, f32> reciprocal(const Simd<8, f32> &a) {
            return Simd<8, f32>(
                1.0f / a.f[0],
                1.0f / a.f[1],
                1.0f / a.f[2],
                1.0f / a.f[3],
                1.0f / a.f[4],
                1.0f / a.f[5],
                1.0f / a.f[6],
                1.0f / a.f[7]);
        }

        static inline Simd<8, f32> cmpEq(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] == b.f[0] ? ~0 : 0;
            r[1] = a.f[1] == b.f[1] ? ~0 : 0;
            r[2] = a.f[2] == b.f[2] ? ~0 : 0;
            r[3] = a.f[3] == b.f[3] ? ~0 : 0;
            r[4] = a.f[4] == b.f[4] ? ~0 : 0;
            r[5] = a.f[5] == b.f[5] ? ~0 : 0;
            r[6] = a.f[6] == b.f[6] ? ~0 : 0;
            r[7] = a.f[7] == b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }
        static inline Simd<8, f32> cmpNe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] != b.f[0] ? ~0 : 0;
            r[1] = a.f[1] != b.f[1] ? ~0 : 0;
            r[2] = a.f[2] != b.f[2] ? ~0 : 0;
            r[3] = a.f[3] != b.f[3] ? ~0 : 0;
            r[4] = a.f[4] != b.f[4] ? ~0 : 0;
            r[5] = a.f[5] != b.f[5] ? ~0 : 0;
            r[6] = a.f[6] != b.f[6] ? ~0 : 0;
            r[7] = a.f[7] != b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }
        static inline Simd<8, f32> cmpLt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] < b.f[0] ? ~0 : 0;
            r[1] = a.f[1] < b.f[1] ? ~0 : 0;
            r[2] = a.f[2] < b.f[2] ? ~0 : 0;
            r[3] = a.f[3] < b.f[3] ? ~0 : 0;
            r[4] = a.f[4] < b.f[4] ? ~0 : 0;
            r[5] = a.f[5] < b.f[5] ? ~0 : 0;
            r[6] = a.f[6] < b.f[6] ? ~0 : 0;
            r[7] = a.f[7] < b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }
        static inline Simd<8, f32> cmpLe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] <= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] <= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] <= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] <= b.f[3] ? ~0 : 0;
            r[4] = a.f[4] <= b.f[4] ? ~0 : 0;
            r[5] = a.f[5] <= b.f[5] ? ~0 : 0;
            r[6] = a.f[6] <= b.f[6] ? ~0 : 0;
            r[7] = a.f[7] <= b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }
        static inline Simd<8, f32> cmpGt(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] > b.f[0] ? ~0 : 0;
            r[1] = a.f[1] > b.f[1] ? ~0 : 0;
            r[2] = a.f[2] > b.f[2] ? ~0 : 0;
            r[3] = a.f[3] > b.f[3] ? ~0 : 0;
            r[4] = a.f[4] > b.f[4] ? ~0 : 0;
            r[5] = a.f[5] > b.f[5] ? ~0 : 0;
            r[6] = a.f[6] > b.f[6] ? ~0 : 0;
            r[7] = a.f[7] > b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }
        static inline Simd<8, f32> cmpGe(const Simd<8, f32> &a, const Simd<8, f32> &b) {
            i32 r[8];
            r[0] = a.f[0] >= b.f[0] ? ~0 : 0;
            r[1] = a.f[1] >= b.f[1] ? ~0 : 0;
            r[2] = a.f[2] >= b.f[2] ? ~0 : 0;
            r[3] = a.f[3] >= b.f[3] ? ~0 : 0;
            r[4] = a.f[4] >= b.f[4] ? ~0 : 0;
            r[5] = a.f[5] >= b.f[5] ? ~0 : 0;
            r[6] = a.f[6] >= b.f[6] ? ~0 : 0;
            r[7] = a.f[7] >= b.f[7] ? ~0 : 0;
            return Simd<8, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7]);
        }

        static inline u32 movemask(const Simd<8, f32> &a) {
            u32 mask = 0;

            mask |= ((*(u32*)&a.f[0] >> 7) & 1) << 0;
            mask |= ((*(u32*)&a.f[0] >> 15) & 1) << 1;
            mask |= ((*(u32*)&a.f[0] >> 23) & 1) << 2;
            mask |= ((*(u32*)&a.f[0] >> 31) & 1) << 3;

            mask |= ((*(u32*)&a.f[1] >> 7) & 1) << 4;
            mask |= ((*(u32*)&a.f[1] >> 15) & 1) << 5;
            mask |= ((*(u32*)&a.f[1] >> 23) & 1) << 6;
            mask |= ((*(u32*)&a.f[1] >> 31) & 1) << 7;

            mask |= ((*(u32*)&a.f[2] >> 7) & 1) << 8;
            mask |= ((*(u32*)&a.f[2] >> 15) & 1) << 9;
            mask |= ((*(u32*)&a.f[2] >> 23) & 1) << 10;
            mask |= ((*(u32*)&a.f[2] >> 31) & 1) << 11;

            mask |= ((*(u32*)&a.f[3] >> 7) & 1) << 12;
            mask |= ((*(u32*)&a.f[3] >> 15) & 1) << 13;
            mask |= ((*(u32*)&a.f[3] >> 23) & 1) << 14;
            mask |= ((*(u32*)&a.f[3] >> 31) & 1) << 15;

            mask |= ((*(u32*)&a.f[4] >> 7) & 1) << 16;
            mask |= ((*(u32*)&a.f[4] >> 15) & 1) << 17;
            mask |= ((*(u32*)&a.f[4] >> 23) & 1) << 18;
            mask |= ((*(u32*)&a.f[4] >> 31) & 1) << 19;

            mask |= ((*(u32*)&a.f[5] >> 7) & 1) << 20;
            mask |= ((*(u32*)&a.f[5] >> 15) & 1) << 21;
            mask |= ((*(u32*)&a.f[5] >> 23) & 1) << 22;
            mask |= ((*(u32*)&a.f[5] >> 31) & 1) << 23;

            mask |= ((*(u32*)&a.f[6] >> 7) & 1) << 24;
            mask |= ((*(u32*)&a.f[6] >> 15) & 1) << 25;
            mask |= ((*(u32*)&a.f[6] >> 23) & 1) << 26;
            mask |= ((*(u32*)&a.f[6] >> 31) & 1) << 27;

            mask |= ((*(u32*)&a.f[7] >> 7) & 1) << 28;
            mask |= ((*(u32*)&a.f[7] >> 15) & 1) << 29;
            mask |= ((*(u32*)&a.f[7] >> 23) & 1) << 30;
            mask |= ((*(u32*)&a.f[7] >> 31) & 1) << 31;
            return mask;
        }
    };

    template <>
    union Simd<16, f32> {
        f32 f[16];
        Simd() {}
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
        static inline void store(f32* ptr, const Simd<16, f32> &simd) {
            ptr[0]  = simd.f[0];
            ptr[1]  = simd.f[1];
            ptr[2]  = simd.f[2];
            ptr[3]  = simd.f[3];
            ptr[4]  = simd.f[4];
            ptr[5]  = simd.f[5];
            ptr[6]  = simd.f[6];
            ptr[7]  = simd.f[7];
            ptr[8]  = simd.f[8];
            ptr[9]  = simd.f[9];
            ptr[10] = simd.f[10];
            ptr[11] = simd.f[11];
            ptr[12] = simd.f[12];
            ptr[13] = simd.f[13];
            ptr[14] = simd.f[14];
            ptr[15] = simd.f[15];
        }
        static inline Simd<16, f32> set1(const f32 value) {
            return Simd<16, f32>(
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
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
        }

        static inline Simd<16, f32> add(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] + b.f[0],
                a.f[1] + b.f[1],
                a.f[2] + b.f[2],
                a.f[3] + b.f[3],
                a.f[4] + b.f[4],
                a.f[5] + b.f[5],
                a.f[6] + b.f[6],
                a.f[7] + b.f[7],
                a.f[8] + b.f[8],
                a.f[9] + b.f[9],
                a.f[10] + b.f[10],
                a.f[11] + b.f[11],
                a.f[12] + b.f[12],
                a.f[13] + b.f[13],
                a.f[14] + b.f[14],
                a.f[15] + b.f[15]);
        }
        static inline Simd<16, f32> sub(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] - b.f[0],
                a.f[1] - b.f[1],
                a.f[2] - b.f[2],
                a.f[3] - b.f[3],
                a.f[4] - b.f[4],
                a.f[5] - b.f[5],
                a.f[6] - b.f[6],
                a.f[7] - b.f[7],
                a.f[8] - b.f[8],
                a.f[9] - b.f[9],
                a.f[10] - b.f[10],
                a.f[11] - b.f[11],
                a.f[12] - b.f[12],
                a.f[13] - b.f[13],
                a.f[14] - b.f[14],
                a.f[15] - b.f[15]);
        }
        static inline Simd<16, f32> neg(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                -a.f[0],
                -a.f[1],
                -a.f[2],
                -a.f[3],
                -a.f[4],
                -a.f[5],
                -a.f[6],
                -a.f[7],
                -a.f[8],
                -a.f[9],
                -a.f[10],
                -a.f[11],
                -a.f[12],
                -a.f[13],
                -a.f[14],
                -a.f[15]);
        }
        static inline Simd<16, f32> abs(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                fabsf(a.f[0]),
                fabsf(a.f[1]),
                fabsf(a.f[2]),
                fabsf(a.f[3]),
                fabsf(a.f[4]),
                fabsf(a.f[5]),
                fabsf(a.f[6]),
                fabsf(a.f[7]),
                fabsf(a.f[8]),
                fabsf(a.f[9]),
                fabsf(a.f[10]),
                fabsf(a.f[11]),
                fabsf(a.f[12]),
                fabsf(a.f[13]),
                fabsf(a.f[14]),
                fabsf(a.f[15]));
        }
        static inline Simd<16, f32> min(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] < b.f[0] ? a.f[0] : b.f[0],
                a.f[1] < b.f[1] ? a.f[1] : b.f[1],
                a.f[2] < b.f[2] ? a.f[2] : b.f[2],
                a.f[3] < b.f[3] ? a.f[3] : b.f[3],
                a.f[4] < b.f[4] ? a.f[4] : b.f[4],
                a.f[5] < b.f[5] ? a.f[5] : b.f[5],
                a.f[6] < b.f[6] ? a.f[6] : b.f[6],
                a.f[7] < b.f[7] ? a.f[7] : b.f[7],
                a.f[8] < b.f[8] ? a.f[8] : b.f[8],
                a.f[9] < b.f[9] ? a.f[9] : b.f[9],
                a.f[10] < b.f[10] ? a.f[10] : b.f[10],
                a.f[11] < b.f[11] ? a.f[11] : b.f[11],
                a.f[12] < b.f[12] ? a.f[12] : b.f[12],
                a.f[13] < b.f[13] ? a.f[13] : b.f[13],
                a.f[14] < b.f[14] ? a.f[14] : b.f[14],
                a.f[15] < b.f[15] ? a.f[15] : b.f[15]);
        }
        static inline Simd<16, f32> max(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] > b.f[0] ? a.f[0] : b.f[0],
                a.f[1] > b.f[1] ? a.f[1] : b.f[1],
                a.f[2] > b.f[2] ? a.f[2] : b.f[2],
                a.f[3] > b.f[3] ? a.f[3] : b.f[3],
                a.f[4] > b.f[4] ? a.f[4] : b.f[4],
                a.f[5] > b.f[5] ? a.f[5] : b.f[5],
                a.f[6] > b.f[6] ? a.f[6] : b.f[6],
                a.f[7] > b.f[7] ? a.f[7] : b.f[7],
                a.f[8] > b.f[8] ? a.f[8] : b.f[8],
                a.f[9] > b.f[9] ? a.f[9] : b.f[9],
                a.f[10] > b.f[10] ? a.f[10] : b.f[10],
                a.f[11] > b.f[11] ? a.f[11] : b.f[11],
                a.f[12] > b.f[12] ? a.f[12] : b.f[12],
                a.f[13] > b.f[13] ? a.f[13] : b.f[13],
                a.f[14] > b.f[14] ? a.f[14] : b.f[14],
                a.f[15] > b.f[15] ? a.f[15] : b.f[15]);
        }

        static inline Simd<16, f32> land(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 ai0  = *(i32*)&a.f[0];
            i32 ai1  = *(i32*)&a.f[1];
            i32 ai2  = *(i32*)&a.f[2];
            i32 ai3  = *(i32*)&a.f[3];
            i32 ai4  = *(i32*)&a.f[4];
            i32 ai5  = *(i32*)&a.f[5];
            i32 ai6  = *(i32*)&a.f[6];
            i32 ai7  = *(i32*)&a.f[7];
            i32 ai8  = *(i32*)&a.f[8];
            i32 ai9  = *(i32*)&a.f[9];
            i32 ai10 = *(i32*)&a.f[10];
            i32 ai11 = *(i32*)&a.f[11];
            i32 ai12 = *(i32*)&a.f[12];
            i32 ai13 = *(i32*)&a.f[13];
            i32 ai14 = *(i32*)&a.f[14];
            i32 ai15 = *(i32*)&a.f[15];
            i32 bi0  = *(i32*)&b.f[0];
            i32 bi1  = *(i32*)&b.f[1];
            i32 bi2  = *(i32*)&b.f[2];
            i32 bi3  = *(i32*)&b.f[3];
            i32 bi4  = *(i32*)&b.f[4];
            i32 bi5  = *(i32*)&b.f[5];
            i32 bi6  = *(i32*)&b.f[6];
            i32 bi7  = *(i32*)&b.f[7];
            i32 bi8  = *(i32*)&b.f[8];
            i32 bi9  = *(i32*)&b.f[9];
            i32 bi10 = *(i32*)&b.f[10];
            i32 bi11 = *(i32*)&b.f[11];
            i32 bi12 = *(i32*)&b.f[12];
            i32 bi13 = *(i32*)&b.f[13];
            i32 bi14 = *(i32*)&b.f[14];
            i32 bi15 = *(i32*)&b.f[15];

            i32 result0  = ai0 & bi0;
            i32 result1  = ai1 & bi1;
            i32 result2  = ai2 & bi2;
            i32 result3  = ai3 & bi3;
            i32 result4  = ai4 & bi4;
            i32 result5  = ai5 & bi5;
            i32 result6  = ai6 & bi6;
            i32 result7  = ai7 & bi7;
            i32 result8  = ai8 & bi8;
            i32 result9  = ai9 & bi9;
            i32 result10 = ai10 & bi10;
            i32 result11 = ai11 & bi11;
            i32 result12 = ai12 & bi12;
            i32 result13 = ai13 & bi13;
            i32 result14 = ai14 & bi14;
            i32 result15 = ai15 & bi15;

            return Simd<16, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7,
                *(f32*)&result8,
                *(f32*)&result9,
                *(f32*)&result10,
                *(f32*)&result11,
                *(f32*)&result12,
                *(f32*)&result13,
                *(f32*)&result14,
                *(f32*)&result15);
        }
        static inline Simd<16, f32> lor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 ai0  = *(i32*)&a.f[0];
            i32 ai1  = *(i32*)&a.f[1];
            i32 ai2  = *(i32*)&a.f[2];
            i32 ai3  = *(i32*)&a.f[3];
            i32 ai4  = *(i32*)&a.f[4];
            i32 ai5  = *(i32*)&a.f[5];
            i32 ai6  = *(i32*)&a.f[6];
            i32 ai7  = *(i32*)&a.f[7];
            i32 ai8  = *(i32*)&a.f[8];
            i32 ai9  = *(i32*)&a.f[9];
            i32 ai10 = *(i32*)&a.f[10];
            i32 ai11 = *(i32*)&a.f[11];
            i32 ai12 = *(i32*)&a.f[12];
            i32 ai13 = *(i32*)&a.f[13];
            i32 ai14 = *(i32*)&a.f[14];
            i32 ai15 = *(i32*)&a.f[15];
            i32 bi0  = *(i32*)&b.f[0];
            i32 bi1  = *(i32*)&b.f[1];
            i32 bi2  = *(i32*)&b.f[2];
            i32 bi3  = *(i32*)&b.f[3];
            i32 bi4  = *(i32*)&b.f[4];
            i32 bi5  = *(i32*)&b.f[5];
            i32 bi6  = *(i32*)&b.f[6];
            i32 bi7  = *(i32*)&b.f[7];
            i32 bi8  = *(i32*)&b.f[8];
            i32 bi9  = *(i32*)&b.f[9];
            i32 bi10 = *(i32*)&b.f[10];
            i32 bi11 = *(i32*)&b.f[11];
            i32 bi12 = *(i32*)&b.f[12];
            i32 bi13 = *(i32*)&b.f[13];
            i32 bi14 = *(i32*)&b.f[14];
            i32 bi15 = *(i32*)&b.f[15];

            i32 result0  = ai0 | bi0;
            i32 result1  = ai1 | bi1;
            i32 result2  = ai2 | bi2;
            i32 result3  = ai3 | bi3;
            i32 result4  = ai4 | bi4;
            i32 result5  = ai5 | bi5;
            i32 result6  = ai6 | bi6;
            i32 result7  = ai7 | bi7;
            i32 result8  = ai8 | bi8;
            i32 result9  = ai9 | bi9;
            i32 result10 = ai10 | bi10;
            i32 result11 = ai11 | bi11;
            i32 result12 = ai12 | bi12;
            i32 result13 = ai13 | bi13;
            i32 result14 = ai14 | bi14;
            i32 result15 = ai15 | bi15;

            return Simd<16, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7,
                *(f32*)&result8,
                *(f32*)&result9,
                *(f32*)&result10,
                *(f32*)&result11,
                *(f32*)&result12,
                *(f32*)&result13,
                *(f32*)&result14,
                *(f32*)&result15);
        }
        static inline Simd<16, f32> lxor(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 ai0  = *(i32*)&a.f[0];
            i32 ai1  = *(i32*)&a.f[1];
            i32 ai2  = *(i32*)&a.f[2];
            i32 ai3  = *(i32*)&a.f[3];
            i32 ai4  = *(i32*)&a.f[4];
            i32 ai5  = *(i32*)&a.f[5];
            i32 ai6  = *(i32*)&a.f[6];
            i32 ai7  = *(i32*)&a.f[7];
            i32 ai8  = *(i32*)&a.f[8];
            i32 ai9  = *(i32*)&a.f[9];
            i32 ai10 = *(i32*)&a.f[10];
            i32 ai11 = *(i32*)&a.f[11];
            i32 ai12 = *(i32*)&a.f[12];
            i32 ai13 = *(i32*)&a.f[13];
            i32 ai14 = *(i32*)&a.f[14];
            i32 ai15 = *(i32*)&a.f[15];
            i32 bi0  = *(i32*)&b.f[0];
            i32 bi1  = *(i32*)&b.f[1];
            i32 bi2  = *(i32*)&b.f[2];
            i32 bi3  = *(i32*)&b.f[3];
            i32 bi4  = *(i32*)&b.f[4];
            i32 bi5  = *(i32*)&b.f[5];
            i32 bi6  = *(i32*)&b.f[6];
            i32 bi7  = *(i32*)&b.f[7];
            i32 bi8  = *(i32*)&b.f[8];
            i32 bi9  = *(i32*)&b.f[9];
            i32 bi10 = *(i32*)&b.f[10];
            i32 bi11 = *(i32*)&b.f[11];
            i32 bi12 = *(i32*)&b.f[12];
            i32 bi13 = *(i32*)&b.f[13];
            i32 bi14 = *(i32*)&b.f[14];
            i32 bi15 = *(i32*)&b.f[15];

            i32 result0  = ai0 ^ bi0;
            i32 result1  = ai1 ^ bi1;
            i32 result2  = ai2 ^ bi2;
            i32 result3  = ai3 ^ bi3;
            i32 result4  = ai4 ^ bi4;
            i32 result5  = ai5 ^ bi5;
            i32 result6  = ai6 ^ bi6;
            i32 result7  = ai7 ^ bi7;
            i32 result8  = ai8 ^ bi8;
            i32 result9  = ai9 ^ bi9;
            i32 result10 = ai10 ^ bi10;
            i32 result11 = ai11 ^ bi11;
            i32 result12 = ai12 ^ bi12;
            i32 result13 = ai13 ^ bi13;
            i32 result14 = ai14 ^ bi14;
            i32 result15 = ai15 ^ bi15;

            return Simd<16, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7,
                *(f32*)&result8,
                *(f32*)&result9,
                *(f32*)&result10,
                *(f32*)&result11,
                *(f32*)&result12,
                *(f32*)&result13,
                *(f32*)&result14,
                *(f32*)&result15);
        }
        static inline Simd<16, f32> lnot(const Simd<16, f32> &a) {
            i32 ai0  = *(i32*)&a.f[0];
            i32 ai1  = *(i32*)&a.f[1];
            i32 ai2  = *(i32*)&a.f[2];
            i32 ai3  = *(i32*)&a.f[3];
            i32 ai4  = *(i32*)&a.f[4];
            i32 ai5  = *(i32*)&a.f[5];
            i32 ai6  = *(i32*)&a.f[6];
            i32 ai7  = *(i32*)&a.f[7];
            i32 ai8  = *(i32*)&a.f[8];
            i32 ai9  = *(i32*)&a.f[9];
            i32 ai10 = *(i32*)&a.f[10];
            i32 ai11 = *(i32*)&a.f[11];
            i32 ai12 = *(i32*)&a.f[12];
            i32 ai13 = *(i32*)&a.f[13];
            i32 ai14 = *(i32*)&a.f[14];
            i32 ai15 = *(i32*)&a.f[15];

            i32 result0  = ~ai0;
            i32 result1  = ~ai1;
            i32 result2  = ~ai2;
            i32 result3  = ~ai3;
            i32 result4  = ~ai4;
            i32 result5  = ~ai5;
            i32 result6  = ~ai6;
            i32 result7  = ~ai7;
            i32 result8  = ~ai8;
            i32 result9  = ~ai9;
            i32 result10 = ~ai10;
            i32 result11 = ~ai11;
            i32 result12 = ~ai12;
            i32 result13 = ~ai13;
            i32 result14 = ~ai14;
            i32 result15 = ~ai15;

            return Simd<16, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7,
                *(f32*)&result8,
                *(f32*)&result9,
                *(f32*)&result10,
                *(f32*)&result11,
                *(f32*)&result12,
                *(f32*)&result13,
                *(f32*)&result14,
                *(f32*)&result15);
        }
        static inline Simd<16, f32> landnot(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 ai0  = *(i32*)&a.f[0];
            i32 ai1  = *(i32*)&a.f[1];
            i32 ai2  = *(i32*)&a.f[2];
            i32 ai3  = *(i32*)&a.f[3];
            i32 ai4  = *(i32*)&a.f[4];
            i32 ai5  = *(i32*)&a.f[5];
            i32 ai6  = *(i32*)&a.f[6];
            i32 ai7  = *(i32*)&a.f[7];
            i32 ai8  = *(i32*)&a.f[8];
            i32 ai9  = *(i32*)&a.f[9];
            i32 ai10 = *(i32*)&a.f[10];
            i32 ai11 = *(i32*)&a.f[11];
            i32 ai12 = *(i32*)&a.f[12];
            i32 ai13 = *(i32*)&a.f[13];
            i32 ai14 = *(i32*)&a.f[14];
            i32 ai15 = *(i32*)&a.f[15];
            i32 bi0  = *(i32*)&b.f[0];
            i32 bi1  = *(i32*)&b.f[1];
            i32 bi2  = *(i32*)&b.f[2];
            i32 bi3  = *(i32*)&b.f[3];
            i32 bi4  = *(i32*)&b.f[4];
            i32 bi5  = *(i32*)&b.f[5];
            i32 bi6  = *(i32*)&b.f[6];
            i32 bi7  = *(i32*)&b.f[7];
            i32 bi8  = *(i32*)&b.f[8];
            i32 bi9  = *(i32*)&b.f[9];
            i32 bi10 = *(i32*)&b.f[10];
            i32 bi11 = *(i32*)&b.f[11];
            i32 bi12 = *(i32*)&b.f[12];
            i32 bi13 = *(i32*)&b.f[13];
            i32 bi14 = *(i32*)&b.f[14];
            i32 bi15 = *(i32*)&b.f[15];

            i32 result0  = ai0 & ~bi0;
            i32 result1  = ai1 & ~bi1;
            i32 result2  = ai2 & ~bi2;
            i32 result3  = ai3 & ~bi3;
            i32 result4  = ai4 & ~bi4;
            i32 result5  = ai5 & ~bi5;
            i32 result6  = ai6 & ~bi6;
            i32 result7  = ai7 & ~bi7;
            i32 result8  = ai8 & ~bi8;
            i32 result9  = ai9 & ~bi9;
            i32 result10 = ai10 & ~bi10;
            i32 result11 = ai11 & ~bi11;
            i32 result12 = ai12 & ~bi12;
            i32 result13 = ai13 & ~bi13;
            i32 result14 = ai14 & ~bi14;
            i32 result15 = ai15 & ~bi15;

            return Simd<16, f32>(
                *(f32*)&result0,
                *(f32*)&result1,
                *(f32*)&result2,
                *(f32*)&result3,
                *(f32*)&result4,
                *(f32*)&result5,
                *(f32*)&result6,
                *(f32*)&result7,
                *(f32*)&result8,
                *(f32*)&result9,
                *(f32*)&result10,
                *(f32*)&result11,
                *(f32*)&result12,
                *(f32*)&result13,
                *(f32*)&result14,
                *(f32*)&result15);
        }

        static inline Simd<16, f32> mul(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] * b.f[0],
                a.f[1] * b.f[1],
                a.f[2] * b.f[2],
                a.f[3] * b.f[3],
                a.f[4] * b.f[4],
                a.f[5] * b.f[5],
                a.f[6] * b.f[6],
                a.f[7] * b.f[7],
                a.f[8] * b.f[8],
                a.f[9] * b.f[9],
                a.f[10] * b.f[10],
                a.f[11] * b.f[11],
                a.f[12] * b.f[12],
                a.f[13] * b.f[13],
                a.f[14] * b.f[14],
                a.f[15] * b.f[15]);
        }
        static inline Simd<16, f32> div(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            return Simd<16, f32>(
                a.f[0] / b.f[0],
                a.f[1] / b.f[1],
                a.f[2] / b.f[2],
                a.f[3] / b.f[3],
                a.f[4] / b.f[4],
                a.f[5] / b.f[5],
                a.f[6] / b.f[6],
                a.f[7] / b.f[7],
                a.f[8] / b.f[8],
                a.f[9] / b.f[9],
                a.f[10] / b.f[10],
                a.f[11] / b.f[11],
                a.f[12] / b.f[12],
                a.f[13] / b.f[13],
                a.f[14] / b.f[14],
                a.f[15] / b.f[15]);
        }
        static inline Simd<16, f32> sqrt(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                sqrtf(a.f[0]),
                sqrtf(a.f[1]),
                sqrtf(a.f[2]),
                sqrtf(a.f[3]),
                sqrtf(a.f[4]),
                sqrtf(a.f[5]),
                sqrtf(a.f[6]),
                sqrtf(a.f[7]),
                sqrtf(a.f[8]),
                sqrtf(a.f[9]),
                sqrtf(a.f[10]),
                sqrtf(a.f[11]),
                sqrtf(a.f[12]),
                sqrtf(a.f[13]),
                sqrtf(a.f[14]),
                sqrtf(a.f[15]));
        }
        static inline Simd<16, f32> rsqrt(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                1.0f / sqrtf(a.f[0]),
                1.0f / sqrtf(a.f[1]),
                1.0f / sqrtf(a.f[2]),
                1.0f / sqrtf(a.f[3]),
                1.0f / sqrtf(a.f[4]),
                1.0f / sqrtf(a.f[5]),
                1.0f / sqrtf(a.f[6]),
                1.0f / sqrtf(a.f[7]),
                1.0f / sqrtf(a.f[8]),
                1.0f / sqrtf(a.f[9]),
                1.0f / sqrtf(a.f[10]),
                1.0f / sqrtf(a.f[11]),
                1.0f / sqrtf(a.f[12]),
                1.0f / sqrtf(a.f[13]),
                1.0f / sqrtf(a.f[14]),
                1.0f / sqrtf(a.f[15]));
        }
        static inline Simd<16, f32> reciprocal(const Simd<16, f32> &a) {
            return Simd<16, f32>(
                1.0f / a.f[0],
                1.0f / a.f[1],
                1.0f / a.f[2],
                1.0f / a.f[3],
                1.0f / a.f[4],
                1.0f / a.f[5],
                1.0f / a.f[6],
                1.0f / a.f[7],
                1.0f / a.f[8],
                1.0f / a.f[9],
                1.0f / a.f[10],
                1.0f / a.f[11],
                1.0f / a.f[12],
                1.0f / a.f[13],
                1.0f / a.f[14],
                1.0f / a.f[15]);
        }

        static inline Simd<16, f32> cmpEq(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] == b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] == b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] == b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] == b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] == b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] == b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] == b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] == b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] == b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] == b.f[9] ? ~0 : 0;
            r[10] = a.f[10] == b.f[10] ? ~0 : 0;
            r[11] = a.f[11] == b.f[11] ? ~0 : 0;
            r[12] = a.f[12] == b.f[12] ? ~0 : 0;
            r[13] = a.f[13] == b.f[13] ? ~0 : 0;
            r[14] = a.f[14] == b.f[14] ? ~0 : 0;
            r[15] = a.f[15] == b.f[15] ? ~0 : 0;

            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }
        static inline Simd<16, f32> cmpNe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] != b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] != b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] != b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] != b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] != b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] != b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] != b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] != b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] != b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] != b.f[9] ? ~0 : 0;
            r[10] = a.f[10] != b.f[10] ? ~0 : 0;
            r[11] = a.f[11] != b.f[11] ? ~0 : 0;
            r[12] = a.f[12] != b.f[12] ? ~0 : 0;
            r[13] = a.f[13] != b.f[13] ? ~0 : 0;
            r[14] = a.f[14] != b.f[14] ? ~0 : 0;
            r[15] = a.f[15] != b.f[15] ? ~0 : 0;

            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }
        static inline Simd<16, f32> cmpLt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] < b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] < b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] < b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] < b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] < b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] < b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] < b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] < b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] < b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] < b.f[9] ? ~0 : 0;
            r[10] = a.f[10] < b.f[10] ? ~0 : 0;
            r[11] = a.f[11] < b.f[11] ? ~0 : 0;
            r[12] = a.f[12] < b.f[12] ? ~0 : 0;
            r[13] = a.f[13] < b.f[13] ? ~0 : 0;
            r[14] = a.f[14] < b.f[14] ? ~0 : 0;
            r[15] = a.f[15] < b.f[15] ? ~0 : 0;
            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }
        static inline Simd<16, f32> cmpLe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] <= b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] <= b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] <= b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] <= b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] <= b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] <= b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] <= b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] <= b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] <= b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] <= b.f[9] ? ~0 : 0;
            r[10] = a.f[10] <= b.f[10] ? ~0 : 0;
            r[11] = a.f[11] <= b.f[11] ? ~0 : 0;
            r[12] = a.f[12] <= b.f[12] ? ~0 : 0;
            r[13] = a.f[13] <= b.f[13] ? ~0 : 0;
            r[14] = a.f[14] <= b.f[14] ? ~0 : 0;
            r[15] = a.f[15] <= b.f[15] ? ~0 : 0;
            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }
        static inline Simd<16, f32> cmpGt(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] > b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] > b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] > b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] > b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] > b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] > b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] > b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] > b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] > b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] > b.f[9] ? ~0 : 0;
            r[10] = a.f[10] > b.f[10] ? ~0 : 0;
            r[11] = a.f[11] > b.f[11] ? ~0 : 0;
            r[12] = a.f[12] > b.f[12] ? ~0 : 0;
            r[13] = a.f[13] > b.f[13] ? ~0 : 0;
            r[14] = a.f[14] > b.f[14] ? ~0 : 0;
            r[15] = a.f[15] > b.f[15] ? ~0 : 0;
            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }
        static inline Simd<16, f32> cmpGe(const Simd<16, f32> &a, const Simd<16, f32> &b) {
            i32 r[16];
            r[0]  = a.f[0] >= b.f[0] ? ~0 : 0;
            r[1]  = a.f[1] >= b.f[1] ? ~0 : 0;
            r[2]  = a.f[2] >= b.f[2] ? ~0 : 0;
            r[3]  = a.f[3] >= b.f[3] ? ~0 : 0;
            r[4]  = a.f[4] >= b.f[4] ? ~0 : 0;
            r[5]  = a.f[5] >= b.f[5] ? ~0 : 0;
            r[6]  = a.f[6] >= b.f[6] ? ~0 : 0;
            r[7]  = a.f[7] >= b.f[7] ? ~0 : 0;
            r[8]  = a.f[8] >= b.f[8] ? ~0 : 0;
            r[9]  = a.f[9] >= b.f[9] ? ~0 : 0;
            r[10] = a.f[10] >= b.f[10] ? ~0 : 0;
            r[11] = a.f[11] >= b.f[11] ? ~0 : 0;
            r[12] = a.f[12] >= b.f[12] ? ~0 : 0;
            r[13] = a.f[13] >= b.f[13] ? ~0 : 0;
            r[14] = a.f[14] >= b.f[14] ? ~0 : 0;
            r[15] = a.f[15] >= b.f[15] ? ~0 : 0;
            return Simd<16, f32>(
                *(f32*)&r[0],
                *(f32*)&r[1],
                *(f32*)&r[2],
                *(f32*)&r[3],
                *(f32*)&r[4],
                *(f32*)&r[5],
                *(f32*)&r[6],
                *(f32*)&r[7],
                *(f32*)&r[8],
                *(f32*)&r[9],
                *(f32*)&r[10],
                *(f32*)&r[11],
                *(f32*)&r[12],
                *(f32*)&r[13],
                *(f32*)&r[14],
                *(f32*)&r[15]);
        }

        static inline u64 movemask(const Simd<16, f32> &a) {
            u64 mask = 0;

            mask |= ((u64)(*(u32*)&a.f[0] >> 7) & 1) << 0;
            mask |= ((u64)(*(u32*)&a.f[0] >> 15) & 1) << 1;
            mask |= ((u64)(*(u32*)&a.f[0] >> 23) & 1) << 2;
            mask |= ((u64)(*(u32*)&a.f[0] >> 31) & 1) << 3;

            mask |= ((u64)(*(u32*)&a.f[1] >> 7) & 1) << 4;
            mask |= ((u64)(*(u32*)&a.f[1] >> 15) & 1) << 5;
            mask |= ((u64)(*(u32*)&a.f[1] >> 23) & 1) << 6;
            mask |= ((u64)(*(u32*)&a.f[1] >> 31) & 1) << 7;

            mask |= ((u64)(*(u32*)&a.f[2] >> 7) & 1) << 8;
            mask |= ((u64)(*(u32*)&a.f[2] >> 15) & 1) << 9;
            mask |= ((u64)(*(u32*)&a.f[2] >> 23) & 1) << 10;
            mask |= ((u64)(*(u32*)&a.f[2] >> 31) & 1) << 11;

            mask |= ((u64)(*(u32*)&a.f[3] >> 7) & 1) << 12;
            mask |= ((u64)(*(u32*)&a.f[3] >> 15) & 1) << 13;
            mask |= ((u64)(*(u32*)&a.f[3] >> 23) & 1) << 14;
            mask |= ((u64)(*(u32*)&a.f[3] >> 31) & 1) << 15;

            mask |= ((u64)(*(u32*)&a.f[4] >> 7) & 1) << 16;
            mask |= ((u64)(*(u32*)&a.f[4] >> 15) & 1) << 17;
            mask |= ((u64)(*(u32*)&a.f[4] >> 23) & 1) << 18;
            mask |= ((u64)(*(u32*)&a.f[4] >> 31) & 1) << 19;

            mask |= ((u64)(*(u32*)&a.f[5] >> 7) & 1) << 20;
            mask |= ((u64)(*(u32*)&a.f[5] >> 15) & 1) << 21;
            mask |= ((u64)(*(u32*)&a.f[5] >> 23) & 1) << 22;
            mask |= ((u64)(*(u32*)&a.f[5] >> 31) & 1) << 23;

            mask |= ((u64)(*(u32*)&a.f[6] >> 7) & 1) << 24;
            mask |= ((u64)(*(u32*)&a.f[6] >> 15) & 1) << 25;
            mask |= ((u64)(*(u32*)&a.f[6] >> 23) & 1) << 26;
            mask |= ((u64)(*(u32*)&a.f[6] >> 31) & 1) << 27;

            mask |= ((u64)(*(u32*)&a.f[7] >> 7) & 1) << 28;
            mask |= ((u64)(*(u32*)&a.f[7] >> 15) & 1) << 29;
            mask |= ((u64)(*(u32*)&a.f[7] >> 23) & 1) << 30;
            mask |= ((u64)(*(u32*)&a.f[7] >> 31) & 1) << 31;

            mask |= ((u64)(*(u32*)&a.f[8] >> 7) & 1) << 32;
            mask |= ((u64)(*(u32*)&a.f[8] >> 15) & 1) << 33;
            mask |= ((u64)(*(u32*)&a.f[8] >> 23) & 1) << 34;
            mask |= ((u64)(*(u32*)&a.f[8] >> 31) & 1) << 35;

            mask |= ((u64)(*(u32*)&a.f[9] >> 7) & 1) << 36;
            mask |= ((u64)(*(u32*)&a.f[9] >> 15) & 1) << 37;
            mask |= ((u64)(*(u32*)&a.f[9] >> 23) & 1) << 38;
            mask |= ((u64)(*(u32*)&a.f[9] >> 31) & 1) << 39;

            mask |= ((u64)(*(u32*)&a.f[10] >> 7) & 1) << 40;
            mask |= ((u64)(*(u32*)&a.f[10] >> 15) & 1) << 41;
            mask |= ((u64)(*(u32*)&a.f[10] >> 23) & 1) << 42;
            mask |= ((u64)(*(u32*)&a.f[10] >> 31) & 1) << 43;

            mask |= ((u64)(*(u32*)&a.f[11] >> 7) & 1) << 44;
            mask |= ((u64)(*(u32*)&a.f[11] >> 15) & 1) << 45;
            mask |= ((u64)(*(u32*)&a.f[11] >> 23) & 1) << 46;
            mask |= ((u64)(*(u32*)&a.f[11] >> 31) & 1) << 47;

            mask |= ((u64)(*(u32*)&a.f[12] >> 7) & 1) << 48;
            mask |= ((u64)(*(u32*)&a.f[12] >> 15) & 1) << 49;
            mask |= ((u64)(*(u32*)&a.f[12] >> 23) & 1) << 50;
            mask |= ((u64)(*(u32*)&a.f[12] >> 31) & 1) << 51;

            mask |= ((u64)(*(u32*)&a.f[13] >> 7) & 1) << 52;
            mask |= ((u64)(*(u32*)&a.f[13] >> 15) & 1) << 53;
            mask |= ((u64)(*(u32*)&a.f[13] >> 23) & 1) << 54;
            mask |= ((u64)(*(u32*)&a.f[13] >> 31) & 1) << 55;

            mask |= ((u64)(*(u32*)&a.f[14] >> 7) & 1) << 56;
            mask |= ((u64)(*(u32*)&a.f[14] >> 15) & 1) << 57;
            mask |= ((u64)(*(u32*)&a.f[14] >> 23) & 1) << 58;
            mask |= ((u64)(*(u32*)&a.f[14] >> 31) & 1) << 59;

            mask |= ((u64)(*(u32*)&a.f[15] >> 7) & 1) << 60;
            mask |= ((u64)(*(u32*)&a.f[15] >> 15) & 1) << 61;
            mask |= ((u64)(*(u32*)&a.f[15] >> 23) & 1) << 62;
            mask |= ((u64)(*(u32*)&a.f[15] >> 31) & 1) << 63;

            return mask;
        }
    };
}

#endif
