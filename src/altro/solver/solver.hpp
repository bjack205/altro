//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "linesearch/linesearch.hpp"
#include "internal_types.hpp"
#include "knotpoint_data.hpp"
#include "shifted_vector.hpp"
#include "solver_options.hpp"
#include "solver_stats.hpp"

namespace altro {

class KnotPointData;

class SolverImpl {
 public:
  explicit SolverImpl(int N);
  ~SolverImpl() = default;

  bool IsInitialized() const { return is_initialized_; }

  ErrorCodes Initialize();
  a_float CalcCost();
//  a_float CalcObjective();
  ErrorCodes OpenLoopRollout();
  ErrorCodes Solve();

  ErrorCodes BackwardPass();
  ErrorCodes MeritFunction(a_float alpha, a_float *phi, a_float *dphi);
  ErrorCodes ForwardPass(a_float *alpha);
  ErrorCodes CopyTrajectory();
  ErrorCodes DualUpdate();
  ErrorCodes PenaltyUpdate();

  ErrorCodes LinearRollout();
  a_float Stationarity();
  a_float Feasibility();

  ErrorCodes CalcConstraints();
  ErrorCodes CalcConstraintJacobians();
  ErrorCodes CalcProjectedDuals();
  ErrorCodes CalcConicJacobians();
  ErrorCodes CalcConicHessians();
  ErrorCodes CalcCostGradient();
  ErrorCodes CalcExpansions();

  // Problem definition
  int horizon_length_;
  std::vector<int> nx_;   // number of states
  std::vector<int> nu_;   // number of inputs
  std::vector<float> h_;  // time steps
  Vector initial_state_;

  // Solver
  AltroOptions opts;
  AltroStats stats;
  std::vector<KnotPointData> data_;

  linesearch::CubicLineSearch ls_;

  // Flags
  bool cost_is_diagonal_ = false;

  // Internal variables for logging
  a_float phi0_;
  a_float dphi0_;
  a_float phi_;
  a_float dphi_;
  a_float rho_;
  int ls_iters_;


 private:
  bool constraint_vals_up_to_date_ = false;
  bool constraint_jacs_up_to_date_ = false;
  bool projected_duals_up_to_date_ = false;
  bool conic_jacs_up_to_date_ = false;
  bool conic_hessians_up_to_date_ = false;
  bool cost_gradients_up_to_date_ = false;
  bool cost_hessians_up_to_date_ = false;
  bool dynamics_jacs_up_to_date_ = false;

  // TVLQR data arrays
  //   Note data is actually stored in data_, these are just pointers to that data to call tvlqr
  std::vector<a_float*> x_;
  std::vector<a_float*> u_;
  std::vector<a_float*> y_;

  std::vector<a_float*> A_;
  std::vector<a_float*> B_;
  std::vector<a_float*> f_;

  std::vector<a_float*> lxx_;
  std::vector<a_float*> luu_;
  std::vector<a_float*> lux_;
  std::vector<a_float*> lx_;
  std::vector<a_float*> lu_;

  std::vector<a_float*> K_;
  std::vector<a_float*> d_;

  std::vector<a_float*> P_;
  std::vector<a_float*> p_;

  std::vector<a_float*> Qxx_;
  std::vector<a_float*> Quu_;
  std::vector<a_float*> Qux_;
  std::vector<a_float*> Qx_;
  std::vector<a_float*> Qu_;

  std::vector<a_float*> Qxx_tmp_;
  std::vector<a_float*> Quu_tmp_;
  std::vector<a_float*> Qux_tmp_;
  std::vector<a_float*> Qx_tmp_;
  std::vector<a_float*> Qu_tmp_;
  a_float delta_V_[2];

  bool is_initialized_ = false;
};

}  // namespace altro
