//
// Created by Brian Jackson on 9/28/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "linesearch.h"

#include <stdbool.h>
#include <math.h>

#include "cubicspline.h"

CubicLineSearch CubicLineSearch_Default() {
  CubicLineSearch result = {
      .max_iters = 25,
      .alpha_max = 2.0,
      .c1 = 1e-4,
      .c2 = 0.9,
      .beta_increase = 1.5,
      .min_interval_size = 1e-6,
      .n_iters = 0,
      .phi0 = 0.0,
      .dphi0 = 0.0,
      .phi_lo = 0.0,
      .phi_hi = 0.0,
      .dphi_lo = 0.0,
      .dphi_hi = 0.0,
      .sufficient_decrease = false,
      .curvature = false,
      .error_code = CLS_NOERROR,
  };
  return result;
}

typedef struct CubicLineSearchFuns {
  MeritFun merit_fun;
  MeritFunDerivative merit_fun_derivative;
  void *merit_fun_thunk;
  void *merit_fun_derivative_thunk;
} CubicLineSearchFuns;

double zoom(CubicLineSearch* const ls, CubicLineSearchFuns* funs, double alo, double ahi);

double CubicLineSearch_Run(CubicLineSearch* const ls, double phi0, double dphi0, double alpha0,
                           MeritFun merit_fun, void* merit_fun_thunk,
                           MeritFunDerivative merit_fun_derivative,
                           void* merit_fun_derivative_thunk, enum CubicLineSearchReturnCodes* err) {
  enum CubicLineSearchReturnCodes err0;
  CubicLineSearchFuns funs = {merit_fun, merit_fun_derivative, merit_fun_thunk, merit_fun_derivative_thunk};
  if (!err) err = &err0;

  if (!ls) {
    *err = CLS_INVALID_POINTER;
    return NAN;
  }

  // Check descent direction
  if (dphi0 <= 0.0) {
    *err = CLS_NOT_DESCENT_DIRECTION;
    return NAN;
  }

  // Store the merit function value and derivative at 0
  ls->phi0 = phi0;
  ls->dphi0 = dphi0;

  // Reset the linesearch struct
  ls->n_iters = 0;
  ls->sufficient_decrease = false;
  ls->curvature = false;
  ls->error_code = CLS_NOERROR;

  bool sufficient_decrease_satisfied = false;
  bool function_not_decreasing = false;
  bool strong_wolfe_satisfied = false;

  /*
   * Stage 1: Increase the interval size until
   *   1. the largest step size has a small gradient
   *   2. the window is guaranteed to contain an acceptable step length
   */
  double c1 = ls->c1;
  double c2 = ls->c2;
  double alpha_prev = alpha0;
  double phi_prev = phi0;
  double alpha = alpha0;
  for (int iter = 0; iter < ls->max_iters; ++iter) {
    double phi = merit_fun(alpha, merit_fun_thunk);
    sufficient_decrease_satisfied = phi <= phi0 + c1 * alpha * dphi0;
    function_not_decreasing = phi >= phi_prev;  // works because phi > phi_prev


    /*
     * Interval contains valid step lengths if either
     *   Case A. The current step violates the sufficient decrease requirement
     *   Case B. phi is trending up
     *
     * Invariants at this point:
     *   - alpha is greater than all points before it
     *   - all previous alphas satisfy the sufficient decrease condition
     */
    if (!sufficient_decrease_satisfied || (iter > 0 && function_not_decreasing)) {
      // call zoom with alo < ahi
      zoom(ls, &funs, alpha_prev, alpha);
    }

    /*
     * Invariants at this point:
     *   - alpha satisfies the sufficient decrease condition
     *   - phi(alpha) is the smallest we've seen so far
     */

    // Check Wolfe conditions
    double dphi = merit_fun_derivative(alpha, merit_fun_derivative_thunk);
    strong_wolfe_satisfied = fabs(dphi) <= -c2 * dphi0;
    if (strong_wolfe_satisfied) {
      ls->sufficient_decrease = true;
      ls->curvature = true;
      ls->error_code = CLS_MINIMUM_FOUND;
      return alpha;
    }

    // Check if gradient is non-negative
    if (dphi >= 0) {
      // Case C
      // We have a "bowl" since the previous gradient was negative
      // alo > ahi
      double alo = alpha;
      double ahi = alpha_prev;
      return zoom(ls, &funs, alo, ahi);
    }

    // Expand the interval
    alpha = alpha * ls->beta_increase;
    if (alpha > ls->alpha_max) {
      // Note: once it hits the end twice, the function_not_decreasing flag will be set
      alpha = ls->alpha_max;
    }

    ls->n_iters += 1;
  }

  return 0;
}

/*
 * Following conditions hold for alo and ahi
 *
 * Their containing interval (amin,amax) satisfies at least one of the following
 * (i) amax violates the sufficient decrease condition (end point is too high) [case A]
 * (ii) phi(amax) >= phi(amin) (end is higher than the start) [case B]
 * (iii) dphi(amax) >= 0 (end points up) [case C]
 *
 * and:
 * (a) their interval contains step lengths that satisfy the strong Wolfe conditions
 * (b) alo satisfies the sufficient decrease condition and has the lowest function value seen so far
 * (c) dphi(alo) * (ahi - alo) < 0 (gradient of lower point is negative if it's on the left, and
 *     positive if it's on the right)
 */
double zoom(CubicLineSearch* const ls, CubicLineSearchFuns* funs, double alo, double ahi) {
  double a_max;
  double a_min;
  double alpha;
  double phi;
  double dphi;
  double phi_lo, dphi_lo;
  double phi_hi, dphi_hi;
  enum CubicSplineReturnCodes cs_err;

  if (!isfinite(alo) || !isfinite(ahi)) {
    ls->error_code = CLS_GOT_NONFINITE_STEP_SIZE;
    return NAN;
  }

  double phi0 = ls->phi0;
  double dphi0 = ls->dphi0;
  double c1 = ls->c1;
  double c2 = ls->c2;

  for (int zoom_iter = ls->n_iters + 1; zoom_iter < ls->max_iters; ++zoom_iter) {
    // Return if the interval gets too small
    if (fabs(alo - ahi) < ls->min_interval_size) {
      alpha = (alo + ahi) / 2.0;  // check at the midpoint
      phi = funs->merit_fun(alpha, funs->merit_fun_thunk);
      dphi = funs->merit_fun_derivative(alpha, funs->merit_fun_derivative_thunk);
      ls->sufficient_decrease = phi <= phi0 + c1 * alpha * dphi0;
      ls->curvature = fabs(dphi) <= -c2 * dphi0;
      if (ls->sufficient_decrease && ls->curvature) {
        ls->error_code = CLS_MINIMUM_FOUND;
      } else {
        ls->error_code = CLS_WINDOW_TOO_SMALL;
      }
    }

    // Get the ends of the interval
    if (alo < ahi) {
      a_min = alo;
      a_max = ahi;
    } else {
      a_min = ahi;
    }

    // Evaluate the merit function and derivatives at the end points
    phi_lo = funs->merit_fun(alo, funs->merit_fun_thunk);
    phi_hi = funs->merit_fun(ahi, funs->merit_fun_thunk);
    dphi_lo = funs->merit_fun_derivative(alo, funs->merit_fun_derivative_thunk);
    dphi_hi = funs->merit_fun_derivative(ahi, funs->merit_fun_derivative_thunk);

    // Create a cubic spline between the end points
    CubicSpline p = CubicSpline_From2Points(alo, phi_lo, dphi_lo, ahi, phi_hi, dphi_hi, &cs_err);
    bool cubic_spline_failed = true;
    if (cs_err == CS_NOERROR) {
      alpha = CubicSpline_ArgMin(&p, &cs_err);
      if (cs_err == CS_FOUND_MINIMUM && isfinite(alpha)) {
        cubic_spline_failed = false;
      }
    }

    // If cubic interpolation fails, try the midpoint
    if (cubic_spline_failed) {
      alpha = (alo + ahi) / 2;
    }

    phi = funs->merit_fun(alpha, funs->merit_fun_thunk);
    dphi = funs->merit_fun_derivative(alpha, funs->merit_fun_derivative_thunk);
    bool sufficient_decrease = phi <= phi0 + c1 * alpha * dphi0;
    bool higher_than_lo = phi > phi_lo;
    bool curvature = fabs(dphi) <= -c2 * dphi0;

    if (sufficient_decrease && curvature) {
      ls->sufficient_decrease = true;
      ls->curvature = true;
      ls->error_code = CLS_MINIMUM_FOUND;
      return alpha;
    }
    if (!sufficient_decrease || higher_than_lo) {
      // Adjusting ahi
      ahi = alpha;
      phi_hi = phi;
      dphi_hi = dphi_hi;
    } else {
      /*
       * Invariants at the current point:
       *   - alpha satisfies the sufficient decrease condition
       *   - phi < phi_lo
       */

      // Pick the endpoint that keeps the "bowl" shape
      bool reset_ahi = dphi * (ahi - alo);
      if (reset_ahi) {
        ahi = alo;
      }
      alo = alpha;
    }

  }
  ls->error_code = CLS_MAX_ITERS;
  return NAN;
}
