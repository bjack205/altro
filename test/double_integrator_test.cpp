#include "Eigen/Dense"
#include "altro/altro.hpp"
#include "altro/utils/formatting.hpp"
#include "gtest/gtest.h"

using namespace altro;

constexpr int dim = 2;

auto dyn = [](double *xnext, const double *x, const double *u, float h) -> void {
  double b = h * h / 2;
  for (int i = 0; i < dim; ++i) {
    xnext[i] = x[i] + x[i + dim] * h + u[i] * b;
    xnext[i + dim] = x[i + dim] + u[i] * h;
  }
  return;
};

auto jac = [](double *jac, const double *x, const double *, float h) -> void {
  Eigen::Map<Eigen::MatrixXd> J(jac, 2 * dim, 3 * dim);
  J.setZero();
  double b = h * h / 2;
  for (int i = 0; i < dim; ++i) {
    J(i, i) = 1.0;
    J(i + dim, i + dim) = 1.0;
    J(i, i + dim) = h;
    J(i, 2 * dim + i) = b;
    J(i + dim, 2 * dim + i) = h;
  }
};

TEST(DoubleIntegrator, DynamicsTest) {
  const int dim = 2;
  const int num_states = 2 * dim;
  const int num_inputs = dim;

  // Define some test inputs
  Eigen::VectorXd xnext(num_states);
  Eigen::VectorXd x(num_states);
  Eigen::VectorXd u(num_inputs);
  Eigen::MatrixXd J(num_states, num_states + num_inputs);
  x << 0.1, 0.2, 0.3, 0.4;
  u << 10.1, -20.4;
  float h = 0.01;
  dyn(xnext.data(), x.data(), u.data(), h);

  // Check the dynamics
  Eigen::VectorXd xn_expected(num_states);
  xn_expected << 0.10350500000000001, 0.20298000000000002, 0.40099999999999997, 0.19600000000000004;
  EXPECT_LT((xnext - xn_expected).norm(), 1e-8);

  // Check the Jacobian
  Eigen::MatrixXd J_expected(num_states, num_states + num_inputs);
  double b = h * h / 2;
  // clang-format off
  J_expected <<
      1, 0, h, 0, b, 0,
      0, 1, 0, h, 0, b,
      0, 0, 1, 0, h, 0,
      0, 0, 0, 1, 0, h;
  // clang-format on
  jac(J.data(), x.data(), u.data(), h);
  EXPECT_LT((J - J_expected).norm(), 1e-8);
}

TEST(DoubleIntegrator, SolverUnconstrained) {
  const int num_horizon = 10;
  const int num_states = 2 * dim;
  const int num_inputs = dim;
  float tf = 5.0;
  const float h = tf / static_cast<double>(num_horizon);

  // Objective
  Eigen::VectorXd Q = Eigen::VectorXd::Constant(num_states, 1.0);
  Eigen::VectorXd R = Eigen::VectorXd::Constant(num_inputs, 1e-2);
  Eigen::VectorXd x0(num_states);
  Eigen::VectorXd xf(num_states);
  Eigen::VectorXd uf(num_inputs);
  x0 << 1.0, 2.0, 0.0, 0.0;
  xf.setZero();
  uf.setZero();

  // Define the problem
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  solver.SetLQRCost(0, 0, Q.data(), R.data(), xf.data(), uf.data(), 0, LastIndex);
  solver.SetInitialState(x0.data(), x0.size());
  solver.Initialize();
  EXPECT_TRUE(solver.IsInitialized());
  fmt::print("Solver Initialized.\n");

  // Set the initial trajectory
  Eigen::VectorXd xinit = x0;
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
  solver.SetState(xinit.data(), xinit.size(), 0, LastIndex);
  solver.SetInput(uinit.data(), uinit.size(), 0, LastIndex);

  // Get Initial Cost
  double cost0 = (x0 - xf).transpose() * Q.asDiagonal() * (x0 - xf);
  //  fmt::print("dx = [{}]\n", x)
  fmt::print("cost0 = {}\n", cost0);

  double cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

//  Eigen::VectorXd xk(num_states);
//  Eigen::VectorXd uk(num_inputs);
//  for (int k = 0; k < num_horizon; ++k) {
//    solver.GetState(xk.data(), k);
//    if (k < num_horizon - 1) {
//      solver.GetInput(uk.data(), k);
//      fmt::print("Index {}: [{}], [{}]\n", k, xk.transpose().eval(), uk.transpose().eval());
//    } else {
//      fmt::print("Index {}: [{}]\n", k, xk.transpose().eval());
//    }
//  }

  // Check that it moved closer to the goal
  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_horizon);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, (x_0 - xf).norm());
}

TEST(DoubleIntegrator, SolveGoalConstraint) {
  const int num_horizon = 10;
  const int num_states = 2 * dim;
  const int num_inputs = dim;
  float tf = 5.0;
  const float h = tf / static_cast<double>(num_horizon);

  // Objective
  Eigen::VectorXd Q = Eigen::VectorXd::Constant(num_states, 1.0);
  Eigen::VectorXd R = Eigen::VectorXd::Constant(num_inputs, 1e-2);
  Eigen::VectorXd x0(num_states);
  Eigen::VectorXd xf(num_states);
  Eigen::VectorXd uf(num_inputs);
  x0 << 1.0, 2.0, 0.0, 0.0;
  xf.setZero();
  uf.setZero();

  // Goal Constraint
  auto goalcon = [num_states, xf](double *c, const double *x, const double *u) {
    int n = num_states;
    (void)u;
    for (int i = 0; i < n; ++i) {
      c[i] = x[i] - xf[i];
    }
  };
  auto goaljac = [num_states, num_inputs](double *jac, const double *x, const double *u) {
    int n = num_states;
    int m = num_inputs;
    (void)x;
    (void)u;
    Eigen::Map<Eigen::MatrixXd> J(jac, n, n + m);
    for (int i = 0; i < n; ++i) {
      J(i, i) = 1.0;
    }
  };

  // Define the problem
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  solver.SetLQRCost(0, 0, Q.data(), R.data(), xf.data(), uf.data(), 0, LastIndex);
  solver.SetInitialState(x0.data(), x0.size());
  solver.SetConstraint(goalcon, goaljac, num_states, ConstraintType::EQUALITY, "Goal constraint",
                       num_horizon, 0, nullptr);
  solver.Initialize();
  EXPECT_TRUE(solver.IsInitialized());
  fmt::print("Solver Initialized.\n");

  // Set the initial trajectory
  Eigen::VectorXd xinit = x0;
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
  solver.SetState(xinit.data(), xinit.size(), 0, LastIndex);
  solver.SetInput(uinit.data(), uinit.size(), 0, LastIndex);

  // Get Initial Cost
  double cost0 = (x0 - xf).transpose() * Q.asDiagonal() * (x0 - xf);
  //  fmt::print("dx = [{}]\n", x)
  fmt::print("cost0 = {}\n", cost0);

  double cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

//  Eigen::VectorXd xk(num_states);
//  Eigen::VectorXd uk(num_inputs);
//  for (int k = 0; k <= num_horizon; ++k) {
//    solver.GetState(xk.data(), k);
//    if (k < num_horizon) {
//      solver.GetInput(uk.data(), k);
//      fmt::print("Index {}: [{}], [{}]\n", k, xk.transpose().eval(), uk.transpose().eval());
//    } else {
//      fmt::print("Index {}: [{}]\n", k, xk.transpose().eval());
//    }
//  }

  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_horizon);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, 1e-4);
}