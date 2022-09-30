//
// Created by Brian Jackson on 9/28/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <stdbool.h>

typedef double (*MeritFun)(double, void*);
typedef double (*MeritFunDerivative)(double, void*);

enum CubicLineSearchReturnCodes {
  CLS_NOERROR,
  CLS_MINIMUM_FOUND,
  CLS_INVALID_POINTER,
  CLS_NOT_DESCENT_DIRECTION,
  CLS_WINDOW_TOO_SMALL,
  CLS_GOT_NONFINITE_STEP_SIZE,
  CLS_MAX_ITERS,
};

typedef struct CubicLineSearch {
  // Options
  int max_iters;
  double alpha_max;
  double c1;
  double c2;
  double beta_increase;
  double min_interval_size;

  // Data
  int n_iters;
  double alpha_prev;
  double phi0;
  double dphi0;
  double phi_lo;
  double phi_hi;
  double dphi_lo;
  double dphi_hi;
  bool sufficient_decrease;
  bool curvature;
  enum CubicLineSearchReturnCodes error_code;
} CubicLineSearch;

CubicLineSearch CubicLineSearch_Default();

double CubicLineSearch_Run(CubicLineSearch* const ls, double phi0, double dphi0, double alpha0,
                           MeritFun merit_fun, void* merit_fun_thunk,
                           MeritFunDerivative merit_fun_derivative,
                           void* merit_fun_derivative_thunk, enum CubicLineSearchReturnCodes* err);
