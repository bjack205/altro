#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"

using Eigen::MatrixXd;
using Eigen::MatrixXd;

using namespace altro;

TEST(Pendulum, DynamicsTest) {
  int n = 2;
  int m = 1;
  float h = 0.05;
  auto dyn = MidpointDynamics(n, m, pendulum_dynamics);
  auto jac = MidpointJacobian(n, m, pendulum_dynamics, pendulum_jacobian);
  Vector xn(n);
  Vector x(n);
  Vector u(m);
  x << 0.1, -0.4;
  u << 1.34;
  pendulum_dynamics(xn.data(), x.data(), u.data());
  fmt::print("xdot = [{}]\n", xn.transpose().eval());

  dyn(xn.data(), x.data(), u.data(), h);
  fmt::print("xn = [{}]\n", xn.transpose().eval());

  Vector xn_expected(n);
  xn_expected << 0.08445158545673655, -0.21395149094594346;
  EXPECT_LT((xn - xn_expected).norm(), 1e-6);

  Matrix J(n, n + m);
  jac(J.data(), x.data(), u.data(), h);
  fmt::print("J:\n{}\n", J);

  Matrix J_expected(n, n + m);
  J_expected << 0.9755975228465564, 0.0495, 0.005000000000000001, -0.967268640223389, 0.9557742592228808, 0.198;
  fmt::print("J:\n{}\n", J_expected);
  EXPECT_LT((J - J_expected).norm(), 1e-6);
}

TEST(Pendulum, Unconstrained) {
  const int n = 2;
  const int m = 1;
  const int N = 50;
  const float tf = 3.0;
  const float h = tf / static_cast<double>(N);

  // Objective
  Vector Qd = Vector::Constant(n, 1e-2);
  Vector Rd = Vector::Constant(m, 1e-3);
  Vector Qdf = Vector::Constant(n, 1e-0);
  Vector x0(n);
  Vector xf(n);
  Vector uf(m);
  x0.setZero();
  xf << M_PI, 0.0;
  uf.setZero();

  // Dynamics
  auto dyn = MidpointDynamics(n, m, pendulum_dynamics);
  auto jac = MidpointJacobian(n, m, pendulum_dynamics, pendulum_jacobian);

  // Define the problem
  ErrorCodes err;
  ALTROSolver solver(N);

  // Dimension and Time step
  err = solver.SetDimension(n, m, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetTimeStep(h, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Dynamics
  err = solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Cost Function
  err = solver.SetLQRCost(n, m, Qd.data(), Rd.data(), xf.data(), uf.data(), 0, N);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetLQRCost(n, m, Qdf.data(), Rd.data(), xf.data(), uf.data(), N);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Initial State
  err = solver.SetInitialState(x0.data(), n);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Initialize solver
  err = solver.Initialize();
  EXPECT_EQ(err, ErrorCodes::NoError);
  EXPECT_TRUE(solver.IsInitialized());

  // Set initial trajectory
  Vector u0 = Vector::Constant(m, 0.1);
  solver.SetInput(u0.data(), m, 0, LastIndex);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 20;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);

  Vector xN(n);
  solver.GetState(xN.data(), N);
  Vector xN_expected(n);
  xN_expected << 3.12099917161669, 0.0011966258762942175;
  double xf_err = (xN - xN_expected).norm();
  EXPECT_LT(xf_err, 1e-5);
  EXPECT_LE(solver.GetIterations(), 10);
}

TEST(Pendulum, GoalConstrained) {
  const int n = 2;
  const int m = 1;
  const int N = 20;
  const float tf = 2.0;
  const float h = tf / static_cast<double>(N);

  // Objective
  Vector Qd = Vector::Constant(n, 1e-2);
  Vector Rd = Vector::Constant(m, 1e-3);
  Vector Qdf = Vector::Constant(n, 1e-0);
  Vector x0(n);
  Vector xf(n);
  Vector uf(m);
  x0.setZero();
  xf << M_PI, 0.0;
  uf.setZero();

  // Dynamics
  auto dyn = MidpointDynamics(n, m, pendulum_dynamics);
  auto jac = MidpointJacobian(n, m, pendulum_dynamics, pendulum_jacobian);

  // Define the problem
  ErrorCodes err;
  ALTROSolver solver(N);

  // Dimension and Time step
  err = solver.SetDimension(n, m, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetTimeStep(h, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Dynamics
  err = solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Cost Function
  err = solver.SetLQRCost(n, m, Qd.data(), Rd.data(), xf.data(), uf.data(), 0, N);
  EXPECT_EQ(err, ErrorCodes::NoError);
  err = solver.SetLQRCost(n, m, Qdf.data(), Rd.data(), xf.data(), uf.data(), N);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Goal Constraint
  auto goal_con = [xf](a_float *c, const a_float *x, const a_float *u) {
    (void)u;
    for (int i = 0; i < xf.size(); ++i) {
      c[i] = xf[i] - x[i];
    }
  };
  auto goal_jac = [=](a_float *jac, const a_float *x, const a_float *u) {
    (void)x;
    (void)u;
    Eigen::Map<MatrixXd> J(jac, n, n + m);
    J.setIdentity();
    J *= -1;
  };
  solver.SetConstraint(goal_con, goal_jac, n, ConstraintType::EQUALITY, "Goal Constraint", N, N + 1);

  // Initial State
  err = solver.SetInitialState(x0.data(), n);
  EXPECT_EQ(err, ErrorCodes::NoError);

  // Initialize solver
  err = solver.Initialize();
  EXPECT_EQ(err, ErrorCodes::NoError);
  EXPECT_TRUE(solver.IsInitialized());

  // Set initial trajectory
  Vector u0 = Vector::Constant(m, 0.1);
  solver.SetInput(u0.data(), m, 0, LastIndex);

  // Solve
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 100;
  solver.SetOptions(opts);
  SolveStatus status = solver.Solve();
  EXPECT_EQ(status, SolveStatus::Success);
  fmt::print("status = {}\n", (int)status);

  Vector xN(n);
  solver.GetState(xN.data(), N);
  double dist_to_goal = (xN - xf).norm();
  fmt::print("distance to goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, 1e-4);
  EXPECT_LE(solver.GetIterations(), 10);
}
