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
  Vector x_dot(n);
  Vector x(n);
  Vector u(m);
  x << 1, 0.5, 15 * M_PI / 180.0, -5 * M_PI / 180.0;
  u << 1.1, 0.2;

  // Continuous Dynamics
  Vector x_dot_expected(4);
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
  Vector Qd = Vector::Constant(n, 1e-2);
  Vector Rd = Vector::Constant(m, 1e-3);
  Vector Qdf = Vector::Constant(n, 1e1);
  Vector x0(n);
  Vector xf(n);
  Vector uf(m);
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
  Vector u0 = Vector::Constant(m, 0.0);
  u0 << 0.5, 0;
  solver.SetInput(u0.data(), m);
  solver.OpenLoopRollout();
  Vector x(n);
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
  Vector xN(n);
  solver.GetState(xN.data(), N);
  PrintVectorRow("xN = ", xN);
  EXPECT_LT((xN - xf).norm(), 1e-2);
}

class BicycleMPC : public ::testing::Test {
 public:
  BicycleMPC() : solver(N) {}
 protected:
  void SetUp() override {
    ErrorCodes err;
    // Objective

    Qd = Vector::Constant(n, 1e-2);
    Rd = Vector::Constant(m, 1e-3);
    Qdf = Vector::Constant(n, 1e1);

    // Dynamics
    auto model_ptr = std::make_shared<BicycleModel>();
    ContinuousDynamicsFunction dyn0 = [model_ptr](double *x_dot, const double *x, const double *u) {
      model_ptr->Dynamics(x_dot, x, u);
    };
    ContinuousDynamicsJacobian jac0 = [model_ptr](double *jac, const double *x, const double *u) {
      model_ptr->Jacobian(jac, x, u);
    };
    dyn = MidpointDynamics(n, m, dyn0);
    jac = MidpointJacobian(n, m, dyn0, jac0);


    // Dimension and Time step
    err = solver.SetDimension(n, m);
    EXPECT_EQ(err, ErrorCodes::NoError);

    // Dynamics
    err = solver.SetExplicitDynamics(dyn, jac);
    EXPECT_EQ(err, ErrorCodes::NoError);

    // Read Reference Trajectory
    int N_ref;
    float t_ref;
    ReadScottyTrajectory(&N_ref, &t_ref, &x_ref, &u_ref);

    // Set time step equal to the reference trajectory
    h = t_ref / static_cast<double>(N_ref);
    err = solver.SetTimeStep(h);
    EXPECT_EQ(err, ErrorCodes::NoError);

    // Set Tracking Cost Function
    for (int k = 0; k <= N; ++k) {
      err = solver.SetLQRCost(n, m, Qd.data(), Rd.data(), x_ref.at(k).data(), u_ref.at(k).data(), k);
      EXPECT_EQ(err, ErrorCodes::NoError);
    }

    // Constraints
    auto steering_angle_con = [](a_float *c, const a_float *x, const a_float *u) {
      (void)u;
      a_float delta_max = 30 * M_PI / 180.0;
      c[0] = x[3] - delta_max;
      c[1] = -delta_max - x[3];
    };
    auto steering_angle_jac = [](a_float *jac, const a_float *x, const a_float *u) {
      (void)x;
      (void)u;
      Eigen::Map<Eigen::Matrix<a_float, 2, 6>> J(jac);
      J.setZero();
      J(0, 3) = 1.0;
      J(1, 3) = -1.0;
    };
    err = solver.SetConstraint(steering_angle_con, steering_angle_jac, 2, ConstraintType::INEQUALITY,
                               "steering angle bound", 0, N + 1);
    EXPECT_EQ(err, ErrorCodes::NoError);

    // Initial State
    err = solver.SetInitialState(x_ref[0].data(), n);
    EXPECT_EQ(err, ErrorCodes::NoError);

    // Initialize Solver
    err = solver.Initialize();
    EXPECT_EQ(err, ErrorCodes::NoError);
    EXPECT_TRUE(solver.IsInitialized());

    // Set Initial Trajectory
    a_float average_speed = u_ref[0][0];
    u0 = Vector::Zero(m);
    u0 << average_speed, 0.0;
    err = solver.SetInput(u0.data(), m);
    EXPECT_EQ(err, ErrorCodes::NoError);
    for (int k = 0; k <= N; ++k) {
      solver.SetState(x_ref.at(k).data(), n, k);
    }
  }

  const int n = BicycleModel::NumStates;
  const int m = BicycleModel::NumInputs;
  const int N = 30;
  float h;

  Vector Qd;
  Vector Rd;
  Vector Qdf;

  ExplicitDynamicsFunction dyn;
  ExplicitDynamicsJacobian jac;

  ALTROSolver solver;

  // Reference Trajectory (the "Scotty Dog")
  std::vector<Eigen::Vector4d> x_ref;
  std::vector<Eigen::Vector2d> u_ref;
  Vector u0;
};

TEST_F(BicycleMPC, Tracking_SingleSolve) {
  a_float cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);
  EXPECT_LT(cost_initial, 1e-6);

  solver.OpenLoopRollout();
  cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 80;
  opts.use_backtracking_linesearch = false;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);
}

TEST_F(BicycleMPC, TrackingMPC_2Solves) {
  // MPC Setup
  int Nsim = 100;
  int mpc_iter = 0;
  Vector q(n);
  Vector u_mpc(m);
  std::vector<Vector> x_sim;
  std::vector<Vector> u_sim;
  x_sim.reserve(Nsim);
  u_sim.reserve(Nsim-1);
  x_sim.emplace_back(x_ref[0]);  // push initial state to the front
  for (int i = 0; i < Nsim - 1; ++i) {
    x_sim.emplace_back(Vector::Zero(n));
    u_sim.emplace_back(Vector::Zero(m));
  }

  // Initial Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 80;
  opts.use_backtracking_linesearch = false;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  // Get control
  solver.GetInput(u_sim[mpc_iter].data(), m);

  // Simulate the system forward
  dyn(x_sim[mpc_iter + 1].data(), x_sim[mpc_iter].data(), u_sim[mpc_iter].data(), h);

  // Set new reference trajectory
  ++mpc_iter;
  a_float c_u = 0.5 * u0.dot(Rd.asDiagonal() * u0);
  a_float c;
  for (int k = 0; k <= N; ++k) {
    const Vector& xk_ref = x_ref[k + mpc_iter];
    q.noalias() = Qd.asDiagonal() * xk_ref;
    q *= -1;
    c = 0.5 * q.dot(xk_ref);
    c *= -1;
    if (k < N) {
      c += c_u;
    }
    solver.UpdateLinearCosts(q.data(), nullptr, c, k);
  }
  solver.SetInitialState(x_sim[mpc_iter].data(), n);
  a_float cost_before_shift = solver.CalcCost();
  solver.ShiftTrajectory();
  a_float cost_after_shift = solver.CalcCost();
  EXPECT_LT(cost_after_shift, cost_before_shift);

  // Solve
  status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);
}
