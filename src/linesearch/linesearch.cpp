//
// Created by Brian Jackson on 9/29/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "linesearch.hpp"

#include <cmath>
#include <iostream>

#include "linesearch.h"

extern "C" {
#include "cubicspline.h"
}

namespace linesearch {

bool CubicLineSearch::SetOptimalityTolerances(double c1, double c2) {
  if (c2 <= c1) return false;
  c1_ = c1;
  c2_ = c2;
  return true;
}

bool CubicLineSearch::SetVerbose(bool verbose) {
  bool verbose_original = verbose_;
  verbose_ = verbose;
  return verbose_original;
}

void CubicLineSearch::GetFinalMeritValues(double* phi, double* dphi) const {
  *phi = phi_;
  *dphi = dphi_;
}

double CubicLineSearch::Run(MeritFun merit_fun, double alpha0, double phi0, double dphi0) {
  // Store the merit function value and derivative at 0
  this->phi0_ = phi0;
  this->dphi0_ = dphi0;

  // Reset the line search
  this->n_iters_ = 0;
  this->sufficient_decrease_ = false;
  this->curvature_ = false;
  this->return_code_ = ReturnCodes::NOERROR;

  // Check descent direction
  if (dphi0 >= 0.0) {
    return_code_ = ReturnCodes::NOT_DESCENT_DIRECTION;
    return 0.0;
  }

  /*
   * Stage 1: Increase the interval size until
   *   1. the largest step size has a small gradient
   *   2. the window is guaranteed to contain an acceptable step length
   */
  double alpha_prev = 0.0;
  double phi_prev = phi0;
  double dphi_prev = dphi0;

  double alpha = alpha0;
  double c1 = this->c1_;
  double c2 = this->c2_;
  bool hit_max_alpha = false;
  double& phi = phi_;
  double& dphi = dphi_;

  if (verbose_)
    std::cout << "Starting Cubic Line Search with\n           "
                 " phi0 = "
              << phi0 << ", dphi0 = " << dphi0 << std::endl;

  for (int iter = 0; iter < this->max_iters; ++iter) {
    this->n_iters_ += 1;
    merit_fun(alpha, &phi, &dphi);

    bool sufficient_decrease_satisfied = phi <= phi0 + c1 * alpha * dphi0;
    bool function_not_decreasing = phi >= phi_prev;  // works because phi > phi_prev
    bool strong_wolfe_satisfied = fabs(dphi) <= -c2 * dphi0;
    if (verbose_)
      std::cout << "  iter = " << iter << ": alpha = " << alpha << ", phi = " << phi
                << ", dphi =" << dphi << ". Armijo? " << sufficient_decrease_satisfied << " Wolfe? "
                << strong_wolfe_satisfied << std::endl;

    // Check convergence
    if (sufficient_decrease_satisfied && strong_wolfe_satisfied) {
      if (verbose_) std::cout << "  Optimal Step Found!\n";
      this->sufficient_decrease_ = true;
      this->curvature_ = true;
      this->return_code_ = ReturnCodes::MINIMUM_FOUND;
      return alpha;
    } else if (iter == 0 && try_cubic_first) {
      // Try a cubic interpolation right away if it fails, otherwise continue
      // TODO: do this before the loop, basically as a way to guess the first value of alpha?
      CubicSplineReturnCodes cs_err;
      CubicSpline p = CubicSpline_From2Points(0, phi0, dphi0, alpha, phi, dphi, &cs_err);
      bool cubic_spline_failed = true;
      double alpha_cubic;
      if (cs_err == CS_NOERROR) {
        alpha_cubic = CubicSpline_ArgMin(&p, &cs_err);
        if (cs_err == CS_FOUND_MINIMUM && std::isfinite(alpha_cubic)) {
          if (verbose_) {
            std::cout << "    Used cubic interpolation on initial interval (0, " << alpha
                      << ") and got alpha = " << alpha_cubic << std::endl;
          }
          cubic_spline_failed = false;
        }
      }
      // If interpolation was successful, try evaluating the new point
      if (!cubic_spline_failed) {
        this->n_iters_ += 1;
        double phi_cubic, dphi_cubic;
        ++iter;
        merit_fun(alpha_cubic, &phi_cubic, &dphi_cubic);
        bool sufficient_decrease_satisfied_cubic = phi_cubic <= phi0 + c1 * alpha_cubic * dphi0;
        bool strong_wolfe_satisfied_cubic = fabs(dphi_cubic) <= -c2 * dphi0;
        if (verbose_)
          std::cout << "  iter = " << iter << ": alpha = " << alpha_cubic << ", phi = " << phi_cubic
                    << ", dphi =" << dphi_cubic << ". Armijo? "
                    << sufficient_decrease_satisfied_cubic << " Wolfe? "
                    << strong_wolfe_satisfied_cubic << std::endl;

        if (sufficient_decrease_satisfied_cubic && strong_wolfe_satisfied_cubic) {
          if (verbose_) std::cout << "  Optimal Step Found!\n";
          this->phi_ = phi_cubic;
          this->dphi_ = dphi_cubic;
          this->sufficient_decrease_ = true;
          this->curvature_ = true;
          this->return_code_ = ReturnCodes::MINIMUM_FOUND;
          return alpha_cubic;
        }
      }
    }

    // Optional: fall back to backtracking line search
    if (this->use_backtracking_linesearch) {
      return SimpleBacktracking(merit_fun, alpha0 * (this->beta_decrease));
    }

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
      if (verbose_) {
        std::cout << "    Zooming with alo < hi:\n";
        std::cout << "      Not Sufficient decrease? " << !sufficient_decrease_satisfied
                  << std::endl;
        std::cout << "      Function not decreasing? " << (iter > 0 && function_not_decreasing)
                  << std::endl;
      }
      double alo = alpha_prev;
      this->phi_lo_ = phi_prev;
      this->dphi_lo_ = dphi_prev;
      double ahi = alpha;
      this->phi_hi_ = phi;
      this->dphi_hi_ = dphi;
      return Zoom(merit_fun, alo, ahi);
    }

    /*
     * Invariants at this point:
     *   - alpha satisfies the sufficient decrease condition
     *   - phi(alpha) is the smallest we've seen so far
     */

    // Check if gradient is non-negative
    if (dphi >= 0) {
      // Case C
      // We have a "bowl" since the previous gradient was negative
      // alo > ahi
      double alo = alpha;
      double ahi = alpha_prev;
      this->phi_lo_ = phi;
      this->dphi_lo_ = dphi;
      this->phi_hi_ = phi_prev;
      this->dphi_hi_ = dphi_prev;

      if (verbose_) {
        std::cout << "    Zooming with ahi < lo (" << ahi << ", " << alo << ")\n";
      }
      return Zoom(merit_fun, alo, ahi);
    }

    // Expand the interval
    alpha_prev = alpha;
    alpha = alpha * this->beta_increase;
    if (alpha > this->alpha_max) {
      // Note: once it hits the end twice, the function_not_decreasing flag will be set
      alpha = this->alpha_max;
      if (hit_max_alpha) {
        return_code_ = ReturnCodes::HIT_MAX_STEPSIZE;
        this->sufficient_decrease_ = sufficient_decrease_satisfied;
        this->curvature_ = strong_wolfe_satisfied;
        return alpha;
      } else {
        hit_max_alpha = true;
      }
    }
    if (verbose_) {
      std::cout << "    Expanding interval to alpha = " << alpha << std::endl;
    }

    phi_prev = phi;
    dphi_prev = dphi;
  }

  return alpha;
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
double CubicLineSearch::Zoom(linesearch::MeritFun merit_fun, double alo, double ahi) {
  double a_max;
  double a_min;
  double alpha = alo;
  double& phi = phi_;
  double& dphi = dphi_;
  enum CubicSplineReturnCodes cs_err;

  if (!std::isfinite(alo) || !std::isfinite(ahi)) {
    this->return_code_ = ReturnCodes::GOT_NONFINITE_STEP_SIZE;
    return 0;
  }

  double c1 = this->c1_;
  double c2 = this->c2_;
  double phi0 = this->phi0_;
  double dphi0 = this->dphi0_;

  double phi_lo = this->phi_lo_;
  double phi_hi = this->phi_hi_;
  double dphi_lo = this->dphi_lo_;
  double dphi_hi = this->dphi_hi_;

  for (int zoom_iter = this->n_iters_ + 1; zoom_iter < this->max_iters; ++zoom_iter) {
    // Return if the interval gets too small
    if (fabs(alo - ahi) < this->min_interval_size) {
      if (verbose_)
        std::cout << "    Window size too small with alo = " << alo << ", ahi = " << ahi
                  << std::endl;
      alpha = (alo + ahi) / 2.0;  // check at the midpoint

      this->n_iters_ += 1;
      merit_fun(alpha, &phi, &dphi);
      this->sufficient_decrease_ = phi <= phi0 + c1 * alpha * dphi0;
      this->curvature_ = fabs(dphi) <= -c2 * dphi0;
      if (this->sufficient_decrease_ && this->curvature_) {
        this->return_code_ = ReturnCodes::MINIMUM_FOUND;
      } else {
        this->return_code_ = ReturnCodes::WINDOW_TOO_SMALL;
      }
      return alpha;
    }

    // Get the ends of the interval
    if (alo < ahi) {
      a_min = alo;
      a_max = ahi;
    } else {
      a_min = ahi;
      a_max = alo;
    }

    // Create a cubic spline between the end points
    CubicSpline p = CubicSpline_From2Points(alo, phi_lo, dphi_lo, ahi, phi_hi, dphi_hi, &cs_err);
    bool cubic_spline_failed = true;
    if (cs_err == CS_NOERROR) {
      alpha = CubicSpline_ArgMin(&p, &cs_err);
      if (cs_err == CS_FOUND_MINIMUM && std::isfinite(alpha)) {
        if (verbose_) {
          std::cout << "    Used cubic interpolation on interval (" << a_min << ", " << a_max
                    << ") and got alpha = " << alpha << std::endl;
        }
        cubic_spline_failed = false;
      }
    }

    // If cubic interpolation fails, try the midpoint
    if (cubic_spline_failed) {
      if (verbose_) std::cout << "    Cubic Interpolation failed. Using midpoint.\n";
      alpha = (alo + ahi) / 2;
    }

    this->n_iters_ += 1;
    merit_fun(alpha, &phi, &dphi);
    bool sufficient_decrease = phi <= phi0 + c1 * alpha * dphi0;
    bool higher_than_lo = phi > phi_lo;
    bool curvature = fabs(dphi) <= -c2 * dphi0;
    if (verbose_)
      std::cout << "  zoom iter = " << zoom_iter << ": alpha = " << alpha << ", phi = " << phi
                << ", dphi =" << dphi << ". Armijo? " << sufficient_decrease << " Wolfe? "
                << curvature << std::endl;

    if (sufficient_decrease && curvature) {
      if (verbose_) std::cout << "  Optimal Step Found!\n";
      this->sufficient_decrease_ = true;
      this->curvature_ = true;
      this->return_code_ = ReturnCodes::MINIMUM_FOUND;
      return alpha;
    }
    if (!sufficient_decrease || higher_than_lo) {
      // Adjusting ahi
      if (verbose_) std::cout << "    Adjusting ahi\n";
      ahi = alpha;
      phi_hi = phi;
      dphi_hi = dphi;
    } else {
      /*
       * Invariants at the current point:
       *   - alpha satisfies the sufficient decrease condition
       *   - phi < phi_lo
       */

      // Pick the endpoint that keeps the "bowl" shape
      bool reset_ahi = dphi * (ahi - alo) <= 0;
      if (reset_ahi) {
        ahi = alo;
        phi_hi = phi_lo;
        dphi_hi = dphi_lo;
        if (verbose_) std::cout << "    Setting ahi = alo. ";
      }
      if (verbose_) std::cout << "    Adjusting alo\n";
      alo = alpha;
      phi_lo = phi;
      dphi_lo = dphi;
    }
  }
  this->return_code_ = ReturnCodes::MAX_ITERATIONS;
  return alpha;
}

const char* CubicLineSearch::StatusToString() {
  switch (return_code_) {
    case ReturnCodes::NOERROR:
      return "No error";
      break;
    case ReturnCodes::MINIMUM_FOUND:
      return "Minimum found";
      break;
    case ReturnCodes::INVALID_POINTER:
      return "Invalid pointer";
      break;
    case ReturnCodes::NOT_DESCENT_DIRECTION:
      return "Not a descent direction";
      break;
    case ReturnCodes::WINDOW_TOO_SMALL:
      return "Window too small";
      break;
    case ReturnCodes::GOT_NONFINITE_STEP_SIZE:
      return "Got non-finite step size";
      break;
    case ReturnCodes::MAX_ITERATIONS:
      return "Hit max iterations";
      break;
    case ReturnCodes::HIT_MAX_STEPSIZE:
      return "Hit max stepsize. Try increasing alpha_max";
      break;
    default:
      return "";
      break;
  };
}

double CubicLineSearch::SimpleBacktracking(MeritFun merit_fun, double alpha0) {
  double alpha = alpha0;
  double c1 = this->c1_;
  double& phi = phi_;

  double phi0 = this->phi0_;
  double dphi0 = this->dphi0_;

  for (int iter = 1; iter < this->max_iters; ++iter) {
    this->n_iters_ += 1;
    merit_fun(alpha, &phi, nullptr);

    bool sufficient_decrease_satisfied = phi <= phi0 + c1 * alpha * dphi0;

    // Check convergence
    if (sufficient_decrease_satisfied) {
      if (verbose_) std::cout << "  Optimal Step Found!\n";
      this->sufficient_decrease_ = true;
      this->curvature_ = true;
      this->return_code_ = ReturnCodes::MINIMUM_FOUND;
      return alpha;
      return 0;
    } else {
      alpha *= this->beta_decrease;
    }
  }
  return alpha;
}

}  // namespace linesearch
