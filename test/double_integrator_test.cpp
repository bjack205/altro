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

TEST(DoubleIntegrator, SolverInit) {
  const int num_horizon = 10;
  const int num_states = 2 * dim;
  const int num_inputs = dim;

  // Objective
  Eigen::VectorXd Q = Eigen::VectorXd::Constant(num_states, 1.0);
  Eigen::VectorXd R = Eigen::VectorXd::Constant(num_inputs, 0.01);
  Eigen::VectorXd x0(num_states);
  Eigen::VectorXd xf(num_states);
  Eigen::VectorXd uf(num_inputs);
  x0 << 1.0, 2.0, 0.0, 0.0;
  xf.setZero();
  uf.setZero();

  // Define the problem
  ALTROSolver solver(num_horizon);
  solver.SetDimension(num_states, num_inputs, 0, num_horizon + 1);
  solver.SetExplicitDynamics(dyn, jac, 0, num_horizon);
  solver.SetLQRCost(Q.data(), R.data(), xf.data(), uf.data(), 0, num_horizon + 1);
  solver.SetInitialState(x0.data(), x0.size());
  solver.Initialize();
  EXPECT_TRUE(solver.IsInitialized());

  // Set the initial trajectory
  Eigen::VectorXd xinit = Eigen::VectorXd::Zero(num_states);
  Eigen::VectorXd uinit = Eigen::VectorXd::Zero(num_inputs);
//  solver.SetState(xinit.data(), xinit.size());
  fmt::print("Solver AssertInitialized.\n");

}