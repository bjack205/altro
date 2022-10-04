//
// Created by Brian Jackson on 10/2/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//
// clang-format on

#pragma once

#include <stdbool.h>

#define TVLQR_SUCCESS -1

typedef double lqr_float;

int tvlqr_TotalMemSize(const int *nx, const int *nu, int num_horizon, bool is_diag);

int tvlqr_BackwardPass(const int *nx, const int *nu, int num_horizon,
                       const lqr_float *const *A, const lqr_float *const *B, const lqr_float *const *f,
                       const lqr_float *const *Q, const lqr_float *const *R, const lqr_float *const* H,
                       const lqr_float *const *q, const lqr_float *const *r, lqr_float reg,
                       lqr_float **K, lqr_float **d,
                       lqr_float **P, lqr_float **p, lqr_float *delta_V,
                       lqr_float **Qxx, lqr_float **Quu, lqr_float **Qux,
                       lqr_float **Qx, lqr_float **Qu,
                       lqr_float **Qxx_tmp, lqr_float **Quu_tmp, lqr_float **Qux_tmp,
                       lqr_float **Qx_tmp, lqr_float **Qu_tmp,
                       bool linear_only_update, bool is_diag);

int tvlqr_ForwardPass(const int *nx, const int *nu, int num_horizon,
                      const lqr_float *const *A, const lqr_float *const *B, const lqr_float *const *f,
                      const lqr_float *const *K, const lqr_float *const *d,
                      const lqr_float *const *P, const lqr_float *const *p,
                      const lqr_float *x0, lqr_float **x, lqr_float **u, lqr_float **y);

// clang-format on
