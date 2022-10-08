//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "solver.hpp"

#include "altro/common/knotpoint.hpp"
#include "altro/utils/formatting.hpp"
#include "altrocpp_interface/altrocpp_interface.hpp"
#include "tvlqr/tvlqr.h"

namespace altro {

SolverImpl::SolverImpl(int N)
    : horizon_length_(N),
      nx_(N + 1, 0),
      nu_(N + 1, 0),
      h_(N),
      opts(),
      stats(),
      problem_(N),
      alsolver_(N),
      x_(N + 1, nullptr),
      u_(N, nullptr),
      y_(N + 1, nullptr),
      A_(N, nullptr),
      B_(N, nullptr),
      f_(N, nullptr),
      lxx_(N + 1, nullptr),
      luu_(N, nullptr),
      lux_(N, nullptr),
      lx_(N + 1, nullptr),
      lu_(N, nullptr),
      K_(N, nullptr),
      d_(N, nullptr),
      P_(N, nullptr),
      p_(N, nullptr),
      Qxx_(N + 1, nullptr),
      Quu_(N, nullptr),
      Qux_(N, nullptr),
      Qx_(N + 1, nullptr),
      Qu_(N, nullptr),
      Qxx_tmp_(N + 1, nullptr),
      Quu_tmp_(N, nullptr),
      Qux_tmp_(N, nullptr),
      Qx_tmp_(N + 1, nullptr),
      Qu_tmp_(N, nullptr) {
  altro::TrajectoryXXd traj(0, 0, N);
  trajectory_ = std::make_shared<altro::TrajectoryXXd>(traj);

  // Initialize knot point data
  for (int k = 0; k <= N; ++k) {
    bool is_terminal = (k == N);
    data_.emplace_back(k, is_terminal);
  }
}

ErrorCodes SolverImpl::Initialize() {
  //  alsolver_.InitializeFromProblem(problem_);
  //
  //  // Initialize the trajectory
  //  using KnotPointXXd = KnotPoint<Eigen::Dynamic, Eigen::Dynamic>;
  //  std::vector<KnotPointXXd> knotpoints;
  //  float t = 0.0;
  //  for (int k = 0; k < horizon_length_ + 1; ++k) {
  //    VectorXd x(nx_[k]);
  //    VectorXd u(nu_[k]);
  //    x.setZero();
  //    u.setZero();
  //    KnotPointXXd z(x, u, t, h_[k]);
  //    knotpoints.push_back(std::move(z));
  //    t += h_[k];  // note this results in roundoff error
  //  }
  //  trajectory_ = std::make_shared<TrajectoryXXd>(knotpoints);
  //  alsolver_.SetTrajectory(trajectory_);
  //
  // Initialize knot point data
  ErrorCodes err;
  for (auto& data : data_) {
    err = data.Initialize();
    if (err != ErrorCodes::NoError) {
      return err;
    }
  }

  // Initialize pointer arrays for TVLQR
  int N = horizon_length_;
  for (int k = 0; k < N; ++k) {
    nx_[k] = data_[k].GetStateDim();
    nu_[k] = data_[k].GetInputDim();
    x_[k] = data_[k].x_.data();
    u_[k] = data_[k].u_.data();
    y_[k] = data_[k].y_.data();

    A_[k] = data_[k].A_.data();
    B_[k] = data_[k].B_.data();
    f_[k] = data_[k].f_.data();

    lxx_[k] = data_[k].lxx_.data();
    luu_[k] = data_[k].luu_.data();
    lux_[k] = data_[k].lux_.data();
    lx_[k] = data_[k].lx_.data();
    lu_[k] = data_[k].lu_.data();

    K_[k] = data_[k].K_.data();
    d_[k] = data_[k].d_.data();

    P_[k] = data_[k].P_.data();
    p_[k] = data_[k].p_.data();

    Qxx_[k] = data_[k].Qxx_.data();
    Quu_[k] = data_[k].Quu_.data();
    Qux_[k] = data_[k].Qux_.data();
    Qx_[k] = data_[k].Qx_.data();
    Qu_[k] = data_[k].Qu_.data();

    Qxx_tmp_[k] = data_[k].Qxx_tmp_.data();
    Quu_tmp_[k] = data_[k].Quu_tmp_.data();
    Qux_tmp_[k] = data_[k].Qux_tmp_.data();
    Qx_tmp_[k] = data_[k].Qx_tmp_.data();
    Qu_tmp_[k] = data_[k].Qu_tmp_.data();
  }
  nx_[N] = data_[N].GetStateDim();
  x_[N] = data_[N].x_.data();
  y_[N] = data_[N].y_.data();
  lxx_[N] = data_[N].lxx_.data();
  lx_[N] = data_[N].lx_.data();
  P_[N] = data_[N].P_.data();
  p_[N] = data_[N].p_.data();
  is_initialized_ = true;

  return ErrorCodes::NoError;
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
    case Verbosity::LineSearch:
      ilqropts.verbose = LogLevel::kInner;
      break;
  }
}

a_float SolverImpl::CalcCost() {
  a_float cost = 0.0;
  for (int k = 0; k <= horizon_length_; ++k) {
    cost += data_[k].CalcCost();
  }
  return cost;
  //  return alsolver_.GetiLQRSolver().Cost();
}

ErrorCodes SolverImpl::OpenLoopRollout() {
  if (!IsInitialized()) return ErrorCodes::SolverNotInitialized;
  data_[0].x_ = initial_state_;
  for (int k = 0; k < horizon_length_; ++k) {
    data_[k].CalcDynamics(data_[k + 1].x_.data());
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
  for (int k = 0; k <= horizon_length_; ++k) {
    KnotPointData& z = data_[k];
    z.CalcDynamicsExpansion();
    z.CalcCostExpansion(true);
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::CalcCostGradient() {
  for (auto z = data_.begin(); z < data_.end(); ++z) {
    z->CalcCostExpansion(true);
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::LinearRollout() {
  tvlqr_ForwardPass(nx_.data(), nu_.data(), horizon_length_, A_.data(), B_.data(), f_.data(),
                    K_.data(), d_.data(), P_.data(), p_.data(), initial_state_.data(), x_.data(),
                    u_.data(), y_.data());
  return ErrorCodes::NoError;
}

a_float SolverImpl::Stationarity() {
  int N = horizon_length_;
  a_float res_x = 0;
  a_float res_u = 0;
  CalcExpansions();  // TODO: avoid this call

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


ErrorCodes SolverImpl::ForwardPass(a_float* alpha) {
  auto phi = [this](double alpha, double* phi, double* dphi) {
    this->MeritFunction(alpha, phi, dphi);
  };
  phi(0.0, &phi0_, &dphi0_);
  if (abs(dphi0_) < opts.tol_meritfun_gradient) {
    *alpha = 0.0;
    return ErrorCodes::MeritFunctionGradientTooSmall;
  }

  ls_.SetVerbose(opts.verbose == Verbosity::LineSearch);
  ls_.try_cubic_first = true;
  *alpha = ls_.Run(phi, 1.0, phi0_, dphi0_);
  ls_.GetFinalMeritValues(&phi_, &dphi_);
  auto res = ls_.GetStatus();
  if (std::isnan(*alpha) || res != linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND) {
    // TODO: Provide a more fine-grained return code
    return ErrorCodes::LineSearchFailed;
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::MeritFunction(a_float alpha, a_float* phi, a_float* dphi) {
  if (phi == nullptr) return ErrorCodes::InvalidPointer;
  int N = horizon_length_;
  bool calc_derivative = dphi != nullptr;
  phi_ = 0;
  dphi_ = 0;

  // Set initial state
  data_[0].x_ = initial_state_;
  data_[0].dx_da_.setZero();

  // Simulate forward to calculate the cost
  for (int k = 0; k < N; ++k) {
    // Compute the control
    KnotPointData& z = data_[k];
    KnotPointData& zn = data_[k + 1];
    Vector dx = z.x_ - z.x;  // TODO: avoid these temporary arrays
    Vector du = -z.K_ * dx + alpha * z.d_;
    z.u_ = z.u + du;
    z.y_ = z.P_ * dx + z.p_;

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
  KnotPointData& z = data_[N];
  double cost = z.CalcCost();
  phi_ += cost;
  Vector dx = z.x_ - z.x;
  z.y_ = z.P_ * dx + z.p_;
  *phi = phi_;

  if (calc_derivative) {
    z.CalcCostGradient();
    dphi_ += z.lx_.dot(z.dx_da_);
    *dphi = dphi_;
  }
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::BackwardPass() {
  int N = horizon_length_;

  a_float reg = 0.0;
  bool linear_only_update = false;
  bool is_diag = false;
  int res = tvlqr_BackwardPass(nx_.data(), nu_.data(), N, A_.data(), B_.data(), f_.data(),
                               lxx_.data(), luu_.data(), lux_.data(), lx_.data(), lu_.data(), reg,
                               K_.data(), d_.data(), P_.data(), p_.data(), delta_V_, Qxx_.data(),
                               Quu_.data(), Qux_.data(), Qx_.data(), Qu_.data(), Qxx_tmp_.data(),
                               Quu_tmp_.data(), Qux_tmp_.data(), Qx_tmp_.data(), Qu_tmp_.data(),
                               linear_only_update, is_diag);
  if (res != TVLQR_SUCCESS) {
    return ErrorCodes::BackwardPassFailed;
  } else {
    return ErrorCodes::NoError;
  }
}

ErrorCodes SolverImpl::Solve() {
  // TODO: Copy options to line search
  ErrorCodes err;

  // Initial rollout
  // TODO: make this a TVLQR rollout with affine terms enabled
  OpenLoopRollout();
  CopyTrajectory();  // make the rolled out trajectory the reference trajectory

  // Start the iterations
  double alpha;
  double cost_initial = CalcCost();
  fmt::print("STARTING ALTRO iLQR SOLVE....\n");
  fmt::print("  Initial Cost: {}\n", cost_initial);
  bool is_converged = false;
  bool stop_iterating = false;

  stats.status = SolveStatus::Unsolved;
  int iter;
  for (iter = 0; iter < opts.iterations_max; ++iter) {
    CalcExpansions();
    BackwardPass();
    err = ForwardPass(&alpha);
    CopyTrajectory();
    a_float stationarity = Stationarity();

    // Use stationarity as the termination criteria
    a_float cost_decrease = phi0_ - phi_;
    if (std::abs(stationarity) < opts.tol_stationarity) {
      is_converged = true;
      stop_iterating = true;
      stats.status = SolveStatus::Success;
    }
    if (!is_converged && err == ErrorCodes::MeritFunctionGradientTooSmall) {
      stats.status = SolveStatus::MeritFunGradientTooSmall;
      stop_iterating = true;
    }

    // Print log
    fmt::print("  iter = {:3d}, phi = {:8.4g} -> {:8.4g} ({:8.4g}), dphi = {:8.4g} -> {:8.4g}, alpha = {:10.4g}, stationarity = {:6.4e}\n",
               iter, phi0_, phi_, cost_decrease, dphi0_, dphi_, alpha, stationarity);

    if (stop_iterating) break;
  }
  if (!is_converged && iter == opts.iterations_max) {
    stats.status = SolveStatus::MaxIterations;
  }
  stats.iterations = iter + 1;
  fmt::print("ALTRO SOLVE FINISHED!\n");
  //  SetCppSolverOptions();
  //  alsolver_.Solve();
  return ErrorCodes::NoError;
}

}  // namespace altro
