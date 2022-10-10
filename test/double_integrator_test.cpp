#include "Eigen/Dense"
#include "altro/altro.hpp"
//#include "altro/eigentypes.hpp"
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

auto jac = [](double *jac, const double *x, const double *u, float h) -> void {
  (void)x;
  (void)u;
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
  ErrorCodes err;
  ALTROSolver solver(num_horizon);
  err = solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  PrintErrorCode(err);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.SetTimeStep(h, 0, LastIndex);
  PrintErrorCode(err);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);
  PrintErrorCode(err);

  err = solver.SetLQRCost(num_states, num_inputs, Q.data(), R.data(), xf.data(), uf.data(), 0,
                          LastIndex);
  PrintErrorCode(err);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.SetInitialState(x0.data(), x0.size());
  PrintErrorCode(err);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.Initialize();
  PrintErrorCode(err);
  EXPECT_EQ(err, ErrorCodes::NoError);
  EXPECT_TRUE(solver.IsInitialized());
  fmt::print("Solver Initialized.\n");

  // Set the initial trajectory
  Eigen::VectorXd xinit = x0;
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
  solver.SetState(xinit.data(), xinit.size(), 0, LastIndex);
  solver.SetInput(uinit.data(), uinit.size(), 0, LastIndex);

  // Get Initial Cost
  double cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 3;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

  fmt::print("\nOptimized Trajectory...\n");
  Eigen::VectorXd xk(num_states);
  Eigen::VectorXd uk(num_inputs);
  Eigen::VectorXd yk(num_states);
  for (int k = 0; k <= num_horizon; ++k) {
    solver.GetState(xk.data(), k);
    solver.GetDualDynamics(yk.data(), k);
    if (k < num_horizon) {
      solver.GetInput(uk.data(), k);
      fmt::print("Index {}: [{}],\t\t [{}],\t\t [{}]\n", k, xk.transpose().eval(),
                 uk.transpose().eval(), yk.transpose().eval());
    } else {
      fmt::print("Index {}: [{}], [{}]\n", k, xk.transpose().eval(), yk.transpose().eval());
    }
  }

  // Check that it moved closer to the goal
  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  err = solver.GetState(x_N.data(), num_horizon);
  PrintErrorCode(err);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("x0 = [{}]\n", x0.transpose().eval());
  fmt::print("xf = [{}]\n", xf.transpose().eval());
  fmt::print("x_0 = [{}]\n", x_0.transpose().eval());
  fmt::print("x_N = [{}]\n", x_N.transpose().eval());
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, (x_0 - xf).norm());
  EXPECT_GT(dist_to_goal, 1e-3);
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
  ErrorCodes err;
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  err = solver.SetLQRCost(num_states, num_inputs, Q.data(), R.data(), xf.data(), uf.data(), 0,
                          LastIndex);
  PrintErrorCode(err);
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
  opts.penalty_scaling = 100;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_horizon);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, 1e-4);
  EXPECT_EQ(solver.GetIterations(), 3);
}

TEST(DoubleIntegrator, ControlBounds) {
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
  x0 << 2.0, 2.0, 0.0, 0.0;
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

  // Control Bound Constraint
  const a_float u_bnd = 1.0;
  auto ubnd_con = [u_bnd](double *c, const double *x, const double *u) {
    (void)x;
    int m = 2;
    double u_bound[2] = {u_bnd, u_bnd};
    for (int i = 0; i < m; ++i) {
      c[i] = u[i] - u_bound[i];
      c[i + m] = -u_bound[i] - u[i];
    }
  };
  auto ubnd_jac = [u_bnd](double *jac, const double *x, const double *u) {
    (void)x;
    (void)u;
    int n = 4;
    int m = 2;
    Eigen::Map<Eigen::MatrixXd> J(jac, 4, 6);
    J.setZero();
    for (int i = 0; i < m; ++i) {
      J(i, n + i) = 1.0;
      J(i + m, n + i) = -1.0;
    }
  };

  // Define the problem
  ErrorCodes err;
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  err = solver.SetLQRCost(num_states, num_inputs, Q.data(), R.data(), xf.data(), uf.data(), 0,
                          LastIndex);
  PrintErrorCode(err);
  solver.SetInitialState(x0.data(), x0.size());
  solver.SetConstraint(goalcon, goaljac, num_states, ConstraintType::EQUALITY, "Goal constraint",
                       num_horizon, 0, nullptr);
  solver.SetConstraint(ubnd_con, ubnd_jac, num_inputs * 2, ConstraintType::INEQUALITY,
                       "Control bounds", 0, num_horizon, nullptr);
  solver.Initialize();
  EXPECT_TRUE(solver.IsInitialized());
  fmt::print("Solver Initialized.\n");

  // Set the initial trajectory
  Eigen::VectorXd xinit = x0;
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
  solver.SetState(xinit.data(), xinit.size(), 0, LastIndex);
  solver.SetInput(uinit.data(), uinit.size(), 0, LastIndex);

  // Get Initial Cost
  double cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.penalty_initial = 100;
  opts.penalty_scaling = 100;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_horizon);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, 1e-4);

  // Check that the controls are saturated
  Eigen::VectorXd u0(2);
  solver.GetInput(u0.data(), 0);
  for (int i = 0; i < num_inputs; ++i) {
    EXPECT_NEAR(u0[i], -u_bnd, 1e-4);
  }

  EXPECT_EQ(solver.GetIterations(), 5);
}

TEST(DoubleIntegrator, SOCControlBounds) {
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
  x0 << 2.0, 2.0, 0.0, 0.0;
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

  // Control Bound Constraint
  const a_float u_bnd = 1.0;
  auto usoc_con = [u_bnd](double *c, const double *x, const double *u) {
    (void)x;
    int m = 2;
    for (int i = 0; i < m; ++i) {
      c[i] = u[i];
    }
    c[m] = u_bnd;
  };
  auto usoc_jac = [u_bnd](double *jac, const double *x, const double *u) {
    (void)x;
    (void)u;
    int n = 4;
    int m = 2;
    Eigen::Map<Eigen::MatrixXd> J(jac, m + 1, 6);
    J.setZero();
    for (int i = 0; i < m; ++i) {
      J(i, n + i) = 1.0;
    }
  };

  // Define the problem
  ErrorCodes err;
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  err = solver.SetLQRCost(num_states, num_inputs, Q.data(), R.data(), xf.data(), uf.data(), 0,
                          LastIndex);
  PrintErrorCode(err);
  solver.SetInitialState(x0.data(), x0.size());
  solver.SetConstraint(goalcon, goaljac, num_states, ConstraintType::EQUALITY, "Goal constraint",
                       num_horizon, 0, nullptr);
  solver.SetConstraint(usoc_con, usoc_jac, num_inputs + 1, ConstraintType::SECOND_ORDER_CONE,
                       "Control bounds", 0, num_horizon, nullptr);
  solver.Initialize();
  EXPECT_TRUE(solver.IsInitialized());
  fmt::print("Solver Initialized.\n");

  // Set the initial trajectory
  Eigen::VectorXd xinit = x0;
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
  solver.SetState(xinit.data(), xinit.size(), 0, LastIndex);
  solver.SetInput(uinit.data(), uinit.size(), 0, LastIndex);

  // Get Initial Cost
  double cost_initial = solver.CalcCost();
  fmt::print("Initial cost = {}\n", cost_initial);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.penalty_initial = 1.0;
  opts.penalty_scaling = 100;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  // Print last state
  double cost_final = solver.CalcCost();
  fmt::print("Final cost = {}\n", cost_final);

  Eigen::VectorXd x_0(num_states);
  Eigen::VectorXd x_N(num_states);
  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_horizon);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, 1e-4);

  // Check that the controls are saturated in the norm
  solver.PrintInputTrajectory();
  Eigen::VectorXd u(2);
  for (int k = 0; k < 3; ++k) {
    solver.GetInput(u.data(), 0);
    EXPECT_NEAR(u.norm(), u_bnd, 1e-2);
  }
  EXPECT_EQ(solver.GetIterations(), 9);
}
