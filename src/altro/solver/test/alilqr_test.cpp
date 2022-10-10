//
// Created by Brian Jackson on 10/10/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace altro {

class SolverImpl_PendulumTest : public ::testing::Test {
 public:
  SolverImpl_PendulumTest() : solver(N) {}
 protected:
  void SetUp() override {}

  void InitializePendulumProblem() {
    h = tf / static_cast<double>(N);

    // Objective
    xf << M_PI, 0.0;

    // Dynamics
    auto dyn = MidpointDynamics(n, m, pendulum_dynamics);
    auto jac = MidpointJacobian(n, m, pendulum_dynamics, pendulum_jacobian);

    // Dimension and Time step
    for (int k = 0; k <= N; ++k) {
      KnotPointData& kp = solver.data_[k];

      kp.SetDimension(n, m);
      if (k < N) {
        kp.SetNextStateDimension(n);
        kp.SetTimestep(h);
        kp.SetDynamics(dyn, jac);

        // Set LQR cost
        Vector q = -(Qd.asDiagonal() * xf);
        Vector r = -(Rd.asDiagonal() * uf);
        a_float c = 0.5 * xf.transpose() * Qd.asDiagonal() * xf;
        c += 0.5 * uf.transpose() * Rd.asDiagonal() * uf;
        kp.SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);

      } else {
        // Set LQR cost
        Vector q = -(Qdf.asDiagonal() * xf);
        Vector r = Vector::Zero(m);
        a_float c = 0.5 * xf.transpose() * Qdf.asDiagonal() * xf;
        kp.SetDiagonalCost(n, m, Qdf.data(), Rd.data(), q.data(), r.data(), c);

        // Goal constraint
        auto goal_con = [](a_float *c, const a_float *x, const a_float *u) {
          (void)u;
          Eigen::Vector2d xf_;
          xf_ << M_PI, 0.0;
          for (int i = 0; i < xf_.size(); ++i) {
            c[i] = xf_[i] - x[i];
          }
        };
        auto goal_jac = [](a_float *jac, const a_float *x, const a_float *u) {
          (void)x;
          (void)u;
          Eigen::Map<Matrix> J(jac, 2, 3);
          J.setIdentity();
          J *= -1;
        };
        kp.SetConstraint(std::move(goal_con), std::move(goal_jac), n, ConstraintType::EQUALITY, "Goal constraint");
      }
    }
    solver.data_[N].SetDimension(n, m);
    solver.data_[N].SetNextStateDimension(n);

    // Initial State
    solver.initial_state_ = x0;

    // Initialize solver
    ErrorCodes err;
    err = solver.Initialize();
    EXPECT_EQ(err, ErrorCodes::NoError);
    EXPECT_TRUE(solver.IsInitialized());

    // Set initial trajectory
    Vector u0 = Vector::Constant(m, 0.1);
    for (int k = 0; k < N; ++k) {
      solver.data_[k].u_ = u0;
    }
    solver.OpenLoopRollout();
  }

  int N = 20;
  int n = 2;
  int m = 1;
  float tf = 2.0;
  float h;

  Vector Qd = Vector::Constant(n, 1e-2);
  Vector Rd = Vector::Constant(m, 1e-3);
  Vector Qdf = Vector::Constant(n, 1e-0);
  Vector x0 = Vector::Zero(n);
  Vector xf = Vector(n);
  Vector uf = Vector::Zero(m);

  SolverImpl solver;
};

TEST_F(SolverImpl_PendulumTest, ALiLQR) {
  fmt::print("\n#############################################\n");
  fmt::print("                AL-iLQR\n");
  fmt::print("#############################################\n");
  InitializePendulumProblem();
  ErrorCodes err;

  PrintVectorRow("xf = ", solver.data_[N].x_);
  solver.CopyTrajectory();
  Vector xf_expected(n);

  // Check initial cost (includes AL penalty terms)
  a_float phi0;
  solver.MeritFunction(0.0, &phi0, nullptr);
  a_float phi0_expected = 10.632455092693577;
  EXPECT_NEAR(phi0, phi0_expected, 1e-3);
  fmt::print("phi0 err = {}\n", phi0 - phi0_expected);

  // Update initial expansions
  for (int k = 0; k <= N; ++k) {
    solver.data_[k].CalcDynamicsExpansion();
    solver.data_[k].CalcConstraints();
    solver.data_[k].CalcConstraintJacobians();
    solver.data_[k].CalcProjectedDuals();
    solver.data_[k].CalcConicJacobians();
    solver.data_[k].CalcCostGradient();
  }

  // First Solve
  Vector &xN = solver.data_[N].x_;
  a_float dist_to_goal = 0.0;
  for (int iter = 0; iter < 6; ++iter) {
    solver.CalcExpansions();
    solver.BackwardPass();
    a_float alpha;
    solver.ls_.SetOptimalityTolerances(1e-4, 0.1);
    err = solver.ForwardPass(&alpha);
    EXPECT_EQ(err, ErrorCodes::NoError);
    a_float stationarity = solver.Stationarity();
    dist_to_goal = (xN - xf).norm();
    fmt::print("iter = {}: s = {}, ", iter, stationarity);
    fmt::print("alpha = {}, ls iters = {}, ", alpha, solver.ls_iters_);
    fmt::print("phi = {}, dphi = {}, ||c|| = {}\n", solver.phi_, solver.dphi_, dist_to_goal);
    solver.CopyTrajectory();
  }
  a_float dist_to_goal0 = dist_to_goal;
  EXPECT_NEAR(dist_to_goal, 0.04186387, 1e-3);

  fmt::print("\nDUAL UPDATE\n");
  solver.DualUpdate();
  solver.PenaltyUpdate();
  for (int k = 0; k <= N; ++k) {
    solver.data_[k].CalcDynamicsExpansion();
    solver.data_[k].CalcConstraints();
    solver.data_[k].CalcConstraintJacobians();
    solver.data_[k].CalcProjectedDuals();
    solver.data_[k].CalcConicJacobians();
    solver.data_[k].CalcCostGradient();
  }

  for (int iter = 0; iter < 6; ++iter) {
    solver.CalcExpansions();
    solver.BackwardPass();
    a_float alpha;
    solver.ls_.SetOptimalityTolerances(1e-4, 0.1);
    err = solver.ForwardPass(&alpha);
    if (err == ErrorCodes::MeritFunctionGradientTooSmall) {
      break;
    }
    EXPECT_EQ(err, ErrorCodes::NoError);
    a_float stationarity = solver.Stationarity();
    dist_to_goal = (xN - xf).norm();
    fmt::print("iter = {}: s = {}, ", iter, stationarity);
    fmt::print("alpha = {}, ", alpha);
    fmt::print("phi = {}, dphi = {}, ||c|| = {}\n", solver.phi_, solver.dphi_, dist_to_goal);
    solver.CopyTrajectory();
  }
  EXPECT_LT(dist_to_goal, dist_to_goal0 / 5);

  fmt::print("\nDUAL UPDATE\n");
  solver.DualUpdate();
  solver.PenaltyUpdate();
  solver.PenaltyUpdate();
  for (int k = 0; k <= N; ++k) {
    solver.data_[k].CalcDynamicsExpansion();
    solver.data_[k].CalcConstraints();
    solver.data_[k].CalcConstraintJacobians();
    solver.data_[k].CalcProjectedDuals();
    solver.data_[k].CalcConicJacobians();
    solver.data_[k].CalcCostGradient();
  }
  for (int iter = 0; iter < 6; ++iter) {
    solver.CalcExpansions();
    solver.BackwardPass();
    a_float alpha;
    solver.ls_.SetOptimalityTolerances(1e-4, 0.1);
    err = solver.ForwardPass(&alpha);
    if (err == ErrorCodes::MeritFunctionGradientTooSmall) {
      break;
    }
    EXPECT_EQ(err, ErrorCodes::NoError);
    a_float stationarity = solver.Stationarity();
    dist_to_goal = (xN - xf).norm();
    fmt::print("iter = {}: s = {}, ", iter, stationarity);
    fmt::print("alpha = {}, ", alpha);
    fmt::print("phi = {}, dphi = {}, ||c|| = {}\n", solver.phi_, solver.dphi_, dist_to_goal);
    solver.CopyTrajectory();
  }
  EXPECT_LT(dist_to_goal, 1e-4);
}

}