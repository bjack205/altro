//
// Created by Brian Jackson on 10/10/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"

using Eigen::MatrixXd;

using namespace altro;

TEST(BicycleTests, Dynamics) {
  BicycleModel model;
  int n = BicycleModel::NumStates;
  int m = BicycleModel::NumInputs;
  VectorXd x_dot(n);
  VectorXd x(n);
  VectorXd u(m);
  x << 1, 0.5, 15 * M_PI / 180.0, -5 * M_PI / 180.0;
  u << 1.1, 0.2;

  // Continuous Dynamics
  VectorXd x_dot_expected(4);
  x_dot_expected << 1.0750584102061864, 0.23291503739549996, -0.03560171424038893, 0.2;
  model.Dynamics(x_dot.data(), x.data(), u.data());
  EXPECT_LT((x_dot - x_dot_expected).norm(), 1e-10);

  // Continuous Jacobian
  MatrixXd J(n, n + m);
  MatrixXd J_expected(n, n + m);
  J_expected << -0.0, -0.0, -0.23291503739549996, -0.1290938153359409, 0.9773258274601694, 0.0, 0.0,
      0.0, 1.0750584102061864, 0.5958541510862063, 0.21174094308681812, 0.0, 0.0, 0.0, 0.0,
      0.409087550891862, -0.03236519476398994, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  model.Jacobian(J.data(), x.data(), u.data());
  EXPECT_LT((J - J_expected).norm(), 1e-6);
}

TEST(BicycleTests, Unconstrained_Turn90) {
  const int n = BicycleModel::NumStates;
  const int m = BicycleModel::NumInputs;
  const int N = 30;
  const float tf = 3.0;
  const float h = tf / static_cast<double>(N);

  // Objective
  VectorXd Qd = VectorXd::Constant(n, 1e-2);
  VectorXd Rd = VectorXd::Constant(m, 1e-3);
  VectorXd Qdf = VectorXd::Constant(n, 1e1);
  VectorXd x0(n);
  VectorXd xf(n);
  VectorXd uf(m);
  x0.setZero();
  xf << 1, 2, M_PI_2, 0.0;
  uf.setZero();

  // Dynamics
  auto model_ptr = std::make_shared<BicycleModel>();
  ContinuousDynamicsFunction dyn0 = [model_ptr](double *x_dot, const double *x, const double *u) {
    model_ptr->Dynamics(x_dot, x, u);
  };
  ContinuousDynamicsJacobian jac0 = [model_ptr](double *jac, const double *x, const double *u) {
    model_ptr->Jacobian(jac, x, u);
  };
  auto dyn = MidpointDynamics(n, m, dyn0);
  auto jac = MidpointJacobian(n, m, dyn0, jac0);

  // Define the problem
  ErrorCodes err;
  ALTROSolver solver(N);

  // Dimension and Time step
  err = solver.SetDimension(n, m);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetTimeStep(h);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Dynamics
  err = solver.SetExplicitDynamics(dyn, jac);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Set Cost Function
  err = solver.SetLQRCost(n, m, Qd.data(), Rd.data(), xf.data(), uf.data(), 0, N);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetLQRCost(n, m, Qdf.data(), Rd.data(), xf.data(), uf.data(), N);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Initial State
  err = solver.SetInitialState(x0.data(), n);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Initial Solver
  err = solver.Initialize();
  EXPECT_EQ(err, ErrorCodes::NoError);
  EXPECT_TRUE(solver.IsInitialized());

  // Set initial trajectory
  VectorXd u0 = VectorXd::Constant(m, 0.0);
  u0 << 0.5, 0;
  solver.SetInput(u0.data(), m);
  solver.OpenLoopRollout();
  VectorXd x(n);
  solver.GetState(x.data(), N);
  PrintVectorRow("xN = ", x);

  // Initial Cost
  a_float cost = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 30;
  opts.use_backtracking_linesearch = true;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  fmt::print("status = {}\n", (int)status);

  // Final state
  VectorXd xN(n);
  solver.GetState(xN.data(), N);
  PrintVectorRow("xN = ", xN);
  EXPECT_LT((xN - xf).norm(), 1e-2);
}
