/**
 * @file Simd.h
 * @brief Portable SIMD API umbrella and backend selector (modern API).
 *
 * This header selects the best available SIMD backend (AVX, SSE, or the
 * scalar `Generic` fallback) as detected via `Quark/System/Simd.h` and
 * re-exports the backend-provided `Proton::Simd<N, T>` specializations.
 *
 * Modern API overview (current usage):
 * - Use `Proton::Simd<N, T>` directly; backends provide concrete
 *   specializations (for example `Simd<4, f32>`, `Simd<8, f32>`, etc.).
 * - Each specialization offers static methods which form the public API:
 *     - Construction/load/store: `load(const T* ptr)`, `store(T* ptr, const Simd&)`,
 *       `set1(T)`, `set(...)`.
 *     - Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `min`, `max`.
 *     - Bitwise: `land`, `lor`, `lxor`, `lnot`, `landnot`.
 *     - Comparison: `cmpEq`, `cmpNe`, `cmpLt`, `cmpLe`, `cmpGt`, `cmpGe`.
 *     - Utilities: `sqrt`, `reciprocal`, `rsqrt`, `movemask`, etc.
 * - Lightweight C++ operator overloads are provided (see `SIMD/Simd.h`) so
 *   you can write natural expressions like `a + b`, `-a`, `a & b`, and
 *   `a == b` where `a` and `b` are `Simd<N, T>` values. These operators
 *   forward to the specialization static methods.
 *
 * Typical usage:
 *   using f4 = Proton::Simd<4, f32>;
 *   f4 a = f4::load(srcA);
 *   f4 b = f4::set1(1.0f);
 *   f4 c = a + b;                // forwards to `f4::add(a,b)`
 *   f4 mask = a == b;            // comparison mask vector
 *   u32 m = f4::movemask(mask);  // extract boolean mask bits
 *   f4::store(dst, c);           // store result
 *
 * Notes:
 * - Comparison results are mask vectors (typically all-bits-one for true);
 *   masks are intended to be combined with bitwise ops and inspected via
 *   `movemask` for branching or conditional selection.
 *
 * See `SIMD/Simd.h` for the `Proton::Simd<>` forward declaration and the
 * operator wrappers that forward to backend implementations.
 */

#ifndef ATOM_PRTN_SIMD_H
#define ATOM_PRTN_SIMD_H

#include <Quark/System/Simd.h>

// IWYU pragma: begin_exports
#if ATOM_SIMD_AVX
#    include "SIMD/AVX/f32.h"
#    include "SIMD/AVX/f64.h"
#    include "SIMD/AVX/i32.h"
#    include "SIMD/AVX/i64.h"
#elif ATOM_SIMD_SSE
#    include "SIMD/SSE/f32.h"
#    include "SIMD/SSE/f64.h"
#    include "SIMD/SSE/i32.h"
#    include "SIMD/SSE/i64.h"
#else
#    include "SIMD/Generic/f32.h"
#    include "SIMD/Generic/f64.h"
#    include "SIMD/Generic/i32.h"
#    include "SIMD/Generic/i64.h"
#endif
// IWYU pragma: end_exports

#endif
