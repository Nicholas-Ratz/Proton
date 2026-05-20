/**
 * @file SIMD/Simd.h
 * @brief Forward-declaration and operator wrappers for `Proton::Simd<>`.
 *
 * Declares the `Simd<N,T>` union template (definitions live in backend
 * headers) and provides thin operator overloads that forward to the
 * backend-provided static methods (e.g. `add`, `sub`, `mul`, `cmpEq`).
 * This keeps user-facing code expressive while allowing optimized
 * backend implementations.
 */

#ifndef ATOM_PRTN_SIMD_SIMD_H
#define ATOM_PRTN_SIMD_SIMD_H

#include <Quark/Types.h>

namespace Proton {
    /// Forward declaration of the portable SIMD union template.
    /// Specializations / definitions live in backend headers.
    template <usize N, typename T>
    union Simd;

    /* Arithmetic operators: forward to backend static methods. */
    template <usize N, typename T>
    inline Simd<N, T> operator+(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::add(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator-(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::sub(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator*(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::mul(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator/(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::div(a, b);
    }

    /* Unary arithmetic. */
    template <usize N, typename T>
    inline Simd<N, T> operator-(const Simd<N, T> &a) {
        return Simd<N, T>::neg(a);
    }

    /* Bitwise logical operators forwarded to backend implementations. */
    template <usize N, typename T>
    inline Simd<N, T> operator&(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::land(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator|(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::lor(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator^(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::lxor(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator~(const Simd<N, T> &a) {
        return Simd<N, T>::lnot(a);
    }

    /* Comparison operators returning a mask vector (backend-defined).
     * Typically a true element is represented by all-bits-one in the
     * corresponding lane; backends choose the exact representation. */
    template <usize N, typename T>
    inline Simd<N, T> operator==(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpEq(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator!=(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpNe(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator<(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpLt(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator<=(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpLe(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator>(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpGt(a, b);
    }
    template <usize N, typename T>
    inline Simd<N, T> operator>=(const Simd<N, T> &a, const Simd<N, T> &b) {
        return Simd<N, T>::cmpGe(a, b);
    }
}

#endif
