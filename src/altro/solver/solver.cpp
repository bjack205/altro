//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "solver.hpp"

#include "altro/common/knotpoint.hpp"
#include "altrocpp_interface/altrocpp_interface.hpp"
#include "altro/utils/formatting.hpp"

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
    KnotPointXXd z(x, u, t, h_[k]);
    knotpoints.push_back(std::move(z));
    t += h_[k];  // note this results in roundoff error
  }
  trajectory_ = std::make_shared<TrajectoryXXd>(knotpoints);
  alsolver_.SetTrajectory(trajectory_);

  return is_initialized_;
}

void SolverImpl::SetCppSolverOptions() {
  SolverOptions& cppopts = alsolver_.GetOptions();
  cppopts.cost_tolerance = opts.tol_cost;
  cppopts.gradient_tolerance = opts.tol_stationarity;
  cppopts.maximum_penalty = opts.penalty_max;
  cppopts.initial_penalty = opts.penalty_initial;
  SolverOptions& ilqropts = alsolver_.GetiLQRSolver().GetOptions();
  switch (opts.verbose) {
    case Verbosity::Silent:
      ilqropts.verbose = LogLevel::kSilent;
      break;
    case Verbosity::Outer:
      ilqropts.verbose = LogLevel::kOuter;
      break;
    case Verbosity::Inner:
      ilqropts.verbose = LogLevel::kInner;
      break;
  }
}

void SolverImpl::Solve() {
  SetCppSolverOptions();
  alsolver_.Solve();
}

a_float SolverImpl::CalcCost() { return alsolver_.GetiLQRSolver().Cost(); }

ErrorCodes SolverImpl::BackwardPass() {
  int N = horizon_length_;

  data_[N].CalcTerminalCostToGo();
  for (int k = N - 1; k >= 0; --k) {
    data_[k].CalcActionValueExpansion(data_[k + 1]);
    data_[k].CalcGains();
    data_[k].CalcCostToGo();
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::LinearRollout() {
  int N = horizon_length_;
  data_[0].x_ = initial_state_;
  for (int k = 0; k < N; ++k) {
    data_[k].y_ = data_[k].P_ * data_[k].x_ + data_[k].p_;
    data_[k].u_ = -data_[k].K_ * data_[k].x_ + data_[k].d_;
    data_[k + 1].x_ = data_[k].A_ * data_[k].x_ + data_[k].B_ * data_[k].u_ + data_[k].f_;
  }
  data_[N].y_ = data_[N].P_ * data_[N].x_ + data_[N].p_;

  return ErrorCodes::NoError;
}

a_float SolverImpl::CalcObjective() {
  a_float J = 0;
  for (int k = 0; k <= horizon_length_; ++k) {
    if (k < horizon_length_) {
    }
  }
  return 0;
}

a_float SolverImpl::Stationarity() {
  int N = horizon_length_;
  a_float res_x = 0;
  a_float res_u = 0;

  for (int k = 0; k < N; ++k) {
    KnotPointData& z = data_[k];
    KnotPointData& zn = data_[k + 1];
    res_x = std::max(res_x, (z.lx_ + z.A_.transpose() * zn.y_ - z.y_).lpNorm<Eigen::Infinity>());
    res_u = std::max(res_u, (z.lu_ + z.B_.transpose() * zn.y_).lpNorm<Eigen::Infinity>());
  }
  KnotPointData& z = data_[N];
  res_x = std::max(res_x, (z.lx_ - z.y_).lpNorm<Eigen::Infinity>());

  return std::max(res_x, res_u);
}

ErrorCodes SolverImpl::CalcCostGradient() {
  for (auto z = data_.begin(); z < data_.end(); ++z) {
    z->CalcCostExpansion(true);
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::MeritFunction(a_float alpha, a_float* phi, a_float* dphi) {
  if (phi == nullptr) return ErrorCodes::InvalidPointer;
  int N = horizon_length_;
  bool calc_derivative = dphi != nullptr;
  a_float phi_ = 0;
  a_float dphi_ = 0;

  // Set initial state
  data_[0].x_ = initial_state_;
  data_[0].dx_da_.setZero();

  // Simulate forward to calculate the cost
  for (int k = 0; k < N; ++k) {
    // Compute the control
    KnotPointData& z = data_[k];
    KnotPointData& zn = data_[k + 1];
    Vector dx = z.x_ - z.x;
    Vector du = -z.K_ * dx + alpha * z.d_;
    z.u_ = z.u + du;

    // Simulate the system forward
    z.CalcDynamics(zn.x_.data());

    // Calculate the cost
    double cost = z.CalcCost();
//    fmt::print("cost {}: {}\n", k, cost);
    phi_ += cost;

    if (calc_derivative) {
      // Calculate gradient of x and u with respect to alpha
      z.CalcDynamicsExpansion();
      z.du_da_ = -z.K_ * z.dx_da_ + z.d_;
      zn.dx_da_ = z.A_ * z.dx_da_ + z.B_ * z.du_da_;

      // Calculate the gradient of the cost with respect to alpha
      z.CalcCostGradient();
      dphi_ += z.lx_.dot(z.dx_da_);
      dphi_ += z.lu_.dot(z.du_da_);
    }
  }

  // Terminal knot point
  double cost = data_[N].CalcCost();
  phi_ += cost;
  *phi = phi_;

  if (calc_derivative) {
    KnotPointData& z = data_[N];
    z.CalcCostGradient();
    dphi_ += z.lx_.dot(z.dx_da_);
    *dphi = dphi_;
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::CopyTrajectory() {
  for (int k = 0; k <= horizon_length_; ++k) {
    data_[k].x = data_[k].x_;
    data_[k].y = data_[k].y_;
    if (k < horizon_length_) {
      data_[k].u = data_[k].u_;
    }
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::CalcExpansions() {
  // TODO: do this in parallel
  for (auto& z : data_) {
    z.CalcDynamicsExpansion();
    z.CalcCostExpansion(true);
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::ForwardPass() {
  return ErrorCodes::MaxConstraintsExceeded;
}

}  // namespace altro
