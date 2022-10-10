//
// Created by Brian Jackson on 9/29/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <functional>

namespace linesearch {

using MeritFun = std::function<void(double,double*,double*)>;

class CubicLineSearch {
 public:
  enum class ReturnCodes {
    NOERROR,
    MINIMUM_FOUND,
    INVALID_POINTER,
    NOT_DESCENT_DIRECTION,
    WINDOW_TOO_SMALL,
    GOT_NONFINITE_STEP_SIZE,
    MAX_ITERATIONS,
    HIT_MAX_STEPSIZE,
  };
  double Run(MeritFun merit_fun, double alpha0, double phi0,
             double dphi0);

  bool SetOptimalityTolerances(double c1, double c2);

  ReturnCodes GetStatus() const { return return_code_; }
  bool SufficientDecreaseSatisfied() const { return sufficient_decrease_; }
  bool CurvatureConditionSatisfied() const { return curvature_; }
  int Iterations() const { return n_iters_; }
  const char* StatusToString();
  bool SetVerbose(bool verbose);
  void GetFinalMeritValues(double *phi, double *dphi) const;


  // Options
  int max_iters = 25;
  double alpha_max = 2.0;
  double beta_increase = 1.5;
  double beta_decrease = 0.5;
  double min_interval_size = 1e-6;
  bool try_cubic_first = false;
  bool use_backtracking_linesearch = false;

 private:
  double Zoom(MeritFun merit_fun, double alo, double ahi);

  double SimpleBacktracking(MeritFun merit_fun, double alpha0);

  ReturnCodes return_code_;
  double c1_ = 1e-4;
  double c2_ = 0.9;
  int n_iters_;
  double phi0_;
  double phi_;
  double phi_lo_;
  double phi_hi_;
  double dphi0_;
  double dphi_;
  double dphi_lo_;
  double dphi_hi_;
  bool sufficient_decrease_;
  bool curvature_;
  bool verbose_ = false;
};

}  // namespace linesearch