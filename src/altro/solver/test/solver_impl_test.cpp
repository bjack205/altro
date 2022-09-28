//
// Created by brian on 9/22/22.
//

#include "altro/solver/solver.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace altro {

TEST(SolverImpl, Constructor) {
  int N = 10;
  SolverImpl solver(N);
  EXPECT_FALSE(solver.data_.begin()->IsTerminalKnotPoint());
  EXPECT_TRUE((solver.data_.end() - 1)->IsTerminalKnotPoint());
}

TEST(SolverImpl, TVLQR_Test) {
  const int dim = 2;
  int n = 2 * dim;
  int m = dim;
  int N = 10;
  float h = 0.01;

  // Equilibrium
  Vector xeq(n);
  Vector ueq(m);
  xeq << 1, 2, 0, 0;
  ueq << 0,0;

  // Initial state
  Vector x0(n);
  x0 << 10.5, -20.5, -4, 5;

  // Calculate A,B,f
  Matrix jac(n, n + m);
  Vector f = Vector::Zero(n);

  discrete_double_integrator_dynamics(f.data(), xeq.data(), ueq.data(), h, dim);
  discrete_double_integrator_jacobian(jac.data(), xeq.data(), ueq.data(), h, dim);
  Matrix A = jac.leftCols(n);
  Matrix B = jac.rightCols(m);
  fmt::print("f = [{}]\n", f.transpose().eval());

  // Cost
  Vector Qd = Vector::Constant(n, 1.1);
  Vector Rd = Vector::Constant(m, 0.1);
  Vector Qfd = Qd * 100;
  Matrix Q = Qd.asDiagonal();
  Matrix R = Rd.asDiagonal();
  Matrix Qf = Qfd.asDiagonal();
  Vector q = Vector::Constant(n, 0.01);
  Vector r = Vector::Constant(m, 0.001);
  float c = 10.5;

  // Initialize solver
  ErrorCodes err;
  SolverImpl solver(N);
  solver.initial_state_ = x0;
  for (int k = 0; k < N; ++k) {
    err = solver.data_[k].SetDimension(n, m);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[k].SetNextStateDimension(n);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[k].SetTimestep(h);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[k].SetLinearDynamics(n, n, m, A.data(), B.data(), f.data());
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[k].SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[k].Initialize();
    EXPECT_EQ(err, ErrorCodes::NoError);
    EXPECT_TRUE(solver.data_[k].IsInitialized());
  }
  err = solver.data_[N].SetDimension(n, 0);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.data_[N].SetDiagonalCost(n, 0, Qfd.data(), nullptr, q.data(), nullptr, 0.0);
  EXPECT_EQ(err, ErrorCodes::NoError);

  err = solver.data_[N].Initialize();
  EXPECT_EQ(err, ErrorCodes::NoError);

  EXPECT_TRUE(solver.data_[N].CostFunctionIsQuadratic());
  EXPECT_TRUE(solver.data_[N].IsInitialized());

  // Calculate backward pass
  solver.BackwardPass();
  Matrix K0_expected(m,n);
  Vector d0_expected(m);
  // clang-format off
  K0_expected << 0.7753129718046554, 0.0, 5.840445640045901,
                 0.0, 0.0, 0.7753129718046554, 0.0, 5.840445640045901;
  d0_expected << -7.634078625343007, -15.256221385516275;
  // clang-format on

  // not sure why they're not matching to machine precision
  double K_err = (solver.data_[0].K_ - K0_expected).lpNorm<Eigen::Infinity>();
  double d_err = (solver.data_[0].d_ - d0_expected).lpNorm<Eigen::Infinity>();
  fmt::print("K error {}\n", K_err);
  fmt::print("d error {}\n", d_err);

  EXPECT_LT((solver.data_[0].K_ - K0_expected).norm(), 1e-6);
  EXPECT_LT((solver.data_[0].d_ - d0_expected).norm(), 1e-6);

  // Linear rollout
  solver.LinearRollout();
  Vector xN(n);
  Vector yN(n);
  xN << 20.165445369740308, -0.13732391651279308, -2.3724421496097037, 2.3113121303468707;
  yN << 2218.2089906714345, -15.09563081640724, -260.9586364570674, 254.2543343381558;
  double x_err = (xN - solver.data_[N].x_).lpNorm<Eigen::Infinity>();
  double y_err = (yN - solver.data_[N].y_).lpNorm<Eigen::Infinity>();
  fmt::print("x error {}\n", x_err);
  fmt::print("y error {}\n", y_err);
  EXPECT_LT(x_err, 1e-6);
  EXPECT_LT(y_err, 1e-5);
}


}  // namespace altro