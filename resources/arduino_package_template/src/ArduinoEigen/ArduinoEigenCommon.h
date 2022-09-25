#pragma once
#ifndef ARDUINO_EIGEN_COMMON_H
#define ARDUINO_EIGEN_COMMON_H

#include "utils/ArxTypeTraits/ArxTypeTraits.h"

#if ARX_HAVE_LIBSTDCPLUSPLUS == 0
#error THIS PLATFORM IS NOT SUPPORTED BECAUSE THERE IS NO STANDARD LIBRARY
#else

// replace conflicting macros...
#ifdef abs
#undef abs
template <typename T>
static constexpr T abs(const T& x) {
    return (x > 0) ? x : -x;
}
#endif
#ifdef round
#undef round
template <typename T>
static constexpr T round(const T& x) {
    return (x >= 0) ? (long)(x + 0.5) : (long)(x - 0.5);
}
#endif
#ifdef B0
#undef B0
static constexpr size_t B0 {0};
#endif
#ifdef B1
#undef B1
static constexpr size_t B1 {1};
#endif
#endif  // ARX_HAVE_LIBSTDCPLUSPLUS == 0

#endif  // ARDUINO_EIGEN_COMMON_H
