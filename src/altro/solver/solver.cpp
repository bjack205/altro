//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "solver.hpp"

#include "altrocpp_interface/altrocpp_interface.hpp"
#include "altro/common/knotpoint.hpp"

namespace altro {

bool SolverImpl::Initialize() {
  alsolver_.InitializeFromProblem(problem_);
  is_initialized_ = true;

  // Initialize the trajectory
  using KnotPointXXd = KnotPoint<Eigen::Dynamic, Eigen::Dynamic>;
  std::vector<KnotPointXXd> knotpoints;
  float t = 0.0;
  for (int k = 0; k < horizon_length_ + 1; ++k) {
    VectorXd x(nx_[k]);
    VectorXd u(nu_[k]);
    x.setZero();
    u.setZero();
    KnotPointXXd z(x, u, t);
    knotpoints.push_back(std::move(z));
    t += h_[k];  // note this results in roundoff error
  }
  trajectory_ = std::make_shared<TrajectoryXXd>(knotpoints);

  return is_initialized_;
}

void SolverImpl::SetCppSolverOptions() {
  SolverOptions& cppopts = alsolver_.GetOptions();
  cppopts.cost_tolerance = opts.tol_cost;
  cppopts.gradient_tolerance = opts.tol_stationarity;
  cppopts.maximum_penalty = opts.penalty_max;
  cppopts.initial_penalty = opts.penalty_initial;
  switch (opts.verbose) {
    case Verbosity::Silent:
      cppopts.verbose = LogLevel::kSilent;
      break;
    case Verbosity::Outer:
      cppopts.verbose = LogLevel::kOuter;
      break;
    case Verbosity::Inner:
      cppopts.verbose = LogLevel::kInner;
      break;
  }
}

void SolverImpl::Solve() {
  alsolver_.SetTrajectory(trajectory_);
  SetCppSolverOptions();
  alsolver_.Solve();
}

}  // namespace altro
