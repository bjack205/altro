//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "solver.hpp"

#include "altro/utils/formatting.hpp"
#include "tvlqr/tvlqr.h"

namespace altro {

SolverImpl::SolverImpl(int N)
    : horizon_length_(N),
      nx_(N + 1, 0),
      nu_(N + 1, 0),
      h_(N),
      opts(),
      stats(),
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
      P_(N + 1, nullptr),
      p_(N + 1, nullptr),
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

  // Initialize knot point data
  for (int k = 0; k <= N; ++k) {
    bool is_terminal = (k == N);
    data_.emplace_back(k, is_terminal);
  }
}

ErrorCodes SolverImpl::Initialize() {
  // Initialize knot point data
  ErrorCodes err;
  for (auto& data : data_) {
    err = data.Initialize();
    if (err != ErrorCodes::NoError) {
      return ALTRO_THROW("Failed to initialize the solver", err);
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


/////////////////////////////////////////////
// Rollouts
/////////////////////////////////////////////
ErrorCodes SolverImpl::OpenLoopRollout() {
  if (!IsInitialized()) return ErrorCodes::SolverNotInitialized;
  data_[0].x_ = initial_state_;
  for (int k = 0; k < horizon_length_; ++k) {
    data_[k].CalcDynamics(data_[k + 1].x_.data());
  }
  constraint_vals_up_to_date_ = false;
  constraint_jacs_up_to_date_ = false;
  projected_duals_up_to_date_ = false;
  conic_jacs_up_to_date_ = false;
  conic_hessians_up_to_date_ = false;
  cost_gradients_up_to_date_ = false;
  cost_hessians_up_to_date_ = false;
  dynamics_jacs_up_to_date_ = false;
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::LinearRollout() {
  tvlqr_ForwardPass(nx_.data(), nu_.data(), horizon_length_, A_.data(), B_.data(), f_.data(),
                    K_.data(), d_.data(), P_.data(), p_.data(), initial_state_.data(), x_.data(),
                    u_.data(), y_.data());
  constraint_vals_up_to_date_ = false;
  constraint_jacs_up_to_date_ = false;
  projected_duals_up_to_date_ = false;
  conic_jacs_up_to_date_ = false;
  conic_hessians_up_to_date_ = false;
  cost_gradients_up_to_date_ = false;
  cost_hessians_up_to_date_ = false;
  dynamics_jacs_up_to_date_ = false;
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

/////////////////////////////////////////////
// Cost Methods
/////////////////////////////////////////////

a_float SolverImpl::CalcCost() {
  a_float cost = 0.0;
//  if (constraint_vals_up_to_date_) fmt::print("    Constraints already up-to-date (CalcCost)\n");
  for (int k = 0; k <= horizon_length_; ++k) {
    data_[k].CalcConstraints();
    a_float cost_k = data_[k].CalcCost();
    cost += cost_k;
  }
  constraint_vals_up_to_date_ = true;
  projected_duals_up_to_date_ = true;
  return cost;
}

ErrorCodes SolverImpl::CalcCostGradient() {
  if (cost_gradients_up_to_date_) fmt::print("    Cost grads already up-to-date (CostGradient)\n");
  if (constraint_jacs_up_to_date_) fmt::print("    Con Jacs already up-to-date (CostGradient)\n");
  if (conic_jacs_up_to_date_) fmt::print("    Conic Jacs already up-to-date (CostGradient)\n");
  for (auto knot_point = data_.begin(); knot_point < data_.end(); ++knot_point) {
    knot_point->CalcCostGradient();
  }
  cost_gradients_up_to_date_ = true;
  constraint_jacs_up_to_date_ = true;
  conic_jacs_up_to_date_ = true;
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::CalcExpansions() {
  // TODO: do this in parallel
  // TODO: don't calculate anything that depends on the initial state?
  if (conic_hessians_up_to_date_) fmt::print("    Conic hessians already up-to-date (CalcAll)\n");
  if (cost_hessians_up_to_date_) fmt::print("    Cost hessians already up-to-date (CalcAll)\n");
  for (int k = 0; k <= horizon_length_; ++k) {
    KnotPointData& knot_point = data_[k];
    knot_point.CalcCostHessian();
  }
  conic_hessians_up_to_date_ = true;
  cost_hessians_up_to_date_ = true;
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Optimality Criteria
/////////////////////////////////////////////

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

a_float SolverImpl::Feasibility() {
  a_float viol = 0;
  for (int k = 0; k <= horizon_length_; ++k) {
    KnotPointData& kp = data_[k];
    viol = std::max(viol, kp.CalcViolations());
  }
  return viol;
}

/////////////////////////////////////////////
// Forward Pass
/////////////////////////////////////////////

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
  ls_iters_ = ls_.Iterations();

  // If a simple backtracking linesearch is used and it has to backtrack,
  // the derivative information isn't calculated during the linesearch
  if (opts.use_backtracking_linesearch && (std::abs(*alpha - 1.0) > 0)) {
    for (int k = 0; k <= horizon_length_; ++k) {
      data_[k].CalcDynamicsExpansion();
      data_[k].CalcConstraintJacobians();
      data_[k].CalcCostGradient();
    }
  }

  if (std::isnan(*alpha) || !(res == linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND ||
                              res == linesearch::CubicLineSearch::ReturnCodes::HIT_MAX_STEPSIZE)) {
    // TODO: Provide a more fine-grained return code
    return ALTRO_THROW(fmt::format("Line search failed with code {}", (int)(res)),
                       ErrorCodes::LineSearchFailed);
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
    KnotPointData& knot_point = data_[k];
    KnotPointData& next_knot_point = data_[k + 1];

    Vector dx = knot_point.x_ - knot_point.x;  // TODO: avoid these temporary arrays
    Vector du = -knot_point.K_ * dx + alpha * knot_point.d_;
    knot_point.u_ = knot_point.u + du;
    knot_point.y_ = knot_point.P_ * dx + knot_point.p_;

    // Simulate the system forward
    knot_point.CalcDynamics(next_knot_point.x_.data());  // invalidates all the knot_point

    // Calculate the cost
    knot_point.CalcConstraints();
    double cost = knot_point.CalcCost();  // updates constraints and projected duals
    phi_ += cost;

    if (calc_derivative) {
      // Calculate gradient of x and u with respect to alpha
      knot_point.CalcDynamicsExpansion();
      knot_point.du_da_ = -knot_point.K_ * knot_point.dx_da_ + knot_point.d_;
      next_knot_point.dx_da_ =
          knot_point.A_ * knot_point.dx_da_ + knot_point.B_ * knot_point.du_da_;

      // Calculate the gradient of the cost with respect to alpha
      knot_point.CalcConstraintJacobians();
      knot_point.CalcCostGradient();
      dphi_ += knot_point.lx_.dot(knot_point.dx_da_);
      dphi_ += knot_point.lu_.dot(knot_point.du_da_);
    }
  }

  // Terminal knot point
  KnotPointData& knot_point = data_[N];
  knot_point.CalcConstraints();
  double cost = knot_point.CalcCost();
  phi_ += cost;
  Vector dx = knot_point.x_ - knot_point.x;
  knot_point.y_ = knot_point.P_ * dx + knot_point.p_;
  *phi = phi_;

  if (calc_derivative) {
    knot_point.CalcConstraintJacobians();
    knot_point.CalcCostGradient();
    dphi_ += knot_point.lx_.dot(knot_point.dx_da_);
    *dphi = dphi_;
  }

  // Changing the trajectory invalidates everything
  constraint_vals_up_to_date_ = false;
  constraint_jacs_up_to_date_ = false;
  projected_duals_up_to_date_ = false;
  conic_jacs_up_to_date_ = false;
  conic_hessians_up_to_date_ = false;
  cost_gradients_up_to_date_ = false;
  cost_hessians_up_to_date_ = false;
  dynamics_jacs_up_to_date_ = false;

  // Set the data that was updated
  constraint_vals_up_to_date_ = true;
  projected_duals_up_to_date_ = true;

  if (calc_derivative) {
    dynamics_jacs_up_to_date_ = true;
    cost_gradients_up_to_date_ = true;
    constraint_jacs_up_to_date_ = true;
    conic_jacs_up_to_date_ = true;
  }
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Backward Pass
/////////////////////////////////////////////
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
  // NOTE: this invalidates only the merit function derivative
  if (res != TVLQR_SUCCESS) {
    return ErrorCodes::BackwardPassFailed;
  } else {
    return ErrorCodes::NoError;
  }
}

/////////////////////////////////////////////
// Outer Loop (AL) Updates
/////////////////////////////////////////////
ErrorCodes SolverImpl::DualUpdate() {
  // Assumes constraint are already up-to-date
  if (!IsInitialized()) return ErrorCodes::SolverNotInitialized;
  for (int k = 0; k <= horizon_length_; ++k) {
    data_[k].DualUpdate();
  }
  projected_duals_up_to_date_ = false;
  conic_jacs_up_to_date_ = false;
  conic_hessians_up_to_date_ = false;
  cost_gradients_up_to_date_ = false;
  cost_hessians_up_to_date_ = false;
  return ErrorCodes::NoError;
}

ErrorCodes SolverImpl::PenaltyUpdate() {
  if (!IsInitialized()) return ErrorCodes::SolverAlreadyInitialized;
  for (int k = 0; k <= horizon_length_; ++k) {
    data_[k].PenaltyUpdate(opts.penalty_scaling, opts.penalty_max);
  }
  rho_ = std::min(rho_ * opts.penalty_scaling, opts.penalty_max);
  projected_duals_up_to_date_ = false;
  conic_jacs_up_to_date_ = false;
  conic_hessians_up_to_date_ = false;
  cost_gradients_up_to_date_ = false;
  cost_hessians_up_to_date_ = false;
  return ErrorCodes::NoError;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Solve
////////////////////////////////////////////////////////////////////////////////////////////////////
ErrorCodes SolverImpl::Solve() {
  // TODO: Copy options to line search
  ErrorCodes err;
  ls_.use_backtracking_linesearch = opts.use_backtracking_linesearch;
  rho_ = opts.penalty_initial;

  // Initial rollout
  // TODO: make this a TVLQR rollout with affine terms enabled
  OpenLoopRollout();
  CopyTrajectory();  // make the rolled out trajectory the reference trajectory
  double cost_initial = CalcCost();
  for (int k = 0; k <= horizon_length_; ++k) {
    data_[k].CalcDynamicsExpansion();
    data_[k].CalcConstraintJacobians();
    data_[k].CalcCostGradient();
    data_[k].SetPenalty(opts.penalty_initial);
  }
  dynamics_jacs_up_to_date_ = true;
  constraint_jacs_up_to_date_ = true;
  conic_jacs_up_to_date_ = true;
  cost_gradients_up_to_date_ = true;

  // Start the iterations
  double alpha;
  if (opts.verbose > Verbosity::Silent) {
    fmt::print("STARTING ALTRO iLQR SOLVE....\n");
    fmt::print("  Initial Cost: {}\n", cost_initial);
  }
  bool is_converged = false;
  bool stop_iterating = false;

  stats.status = SolveStatus::Unsolved;
  int iter;
  for (iter = 0; iter < opts.iterations_max; ++iter) {
    CalcExpansions();
    BackwardPass();
    err = ForwardPass(&alpha);
    if (!(err == ErrorCodes::NoError || err == ErrorCodes::MeritFunctionGradientTooSmall)) {
      PrintErrorCode(err);
      stop_iterating = true;
    }

    // Calculate convergence criteria
    // TODO: Add complimentarity?
    // TODO: Add full nonlinear stationarity?
    a_float stationarity = Stationarity();
    a_float feasibility = Feasibility();
    CopyTrajectory();

    a_float cost_decrease = phi0_ - phi_;
    if (std::abs(stationarity) < opts.tol_stationarity &&
        feasibility < opts.tol_primal_feasibility) {
      is_converged = true;
      stop_iterating = true;
      stats.status = SolveStatus::Success;
    }

    // Check if the duals should be updated
    bool dual_update = false;
    a_float penalty = rho_;  // cache this because it can get updated between here and printing
    if (stationarity < std::sqrt(opts.tol_stationarity)) {
      DualUpdate();

      if (feasibility > opts.tol_primal_feasibility) {
        PenaltyUpdate();  // TODO: Maybe do this if the dual update didn't improve the feasibility?
      }

      // Update the projected duals given the updated duals, penalty
      // NOTE: uses constraint values cached during the forward pass
      for (int k = 0; k <= horizon_length_; ++k) {
        data_[k].CalcProjectedDuals();
        data_[k].CalcCostGradient();
      }
      projected_duals_up_to_date_ = true;
      dual_update = true;
    }

    // Print log
    if (opts.verbose > Verbosity::Silent) {
      fmt::print(
          "  iter = {:3d}, phi = {:8.4g} -> {:8.4g} ({:10.3g}), dphi = {:10.3g} -> {:10.3g}, alpha = "
          "{:8.3g}, ls_iter = {:2d}, stat = {:8.3e}, feas = {:8.3e}, rho = {:7.2g}, dual update? "
          "{}\n",
          iter, phi0_, phi_, cost_decrease, dphi0_, dphi_, alpha, ls_iters_, stationarity,
          feasibility, penalty, dual_update);
    }

    if (stop_iterating) break;
  }
  if (!is_converged && iter == opts.iterations_max) {
    stats.status = SolveStatus::MaxIterations;
  }
  stats.iterations = iter + 1;
  if (opts.verbose > Verbosity::Silent) {
    fmt::print("ALTRO SOLVE FINISHED!\n");
  }
  return ErrorCodes::NoError;
}

}  // namespace altro
