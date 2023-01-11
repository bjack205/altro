//
// Created by Brian Jackson on 10/4/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//
#define EIGEN_RUNTIME_NO_MALLOC

#include "tvlqr/tvlqr.h"

#include "Eigen/Dense"
#include "altro/utils/formatting.hpp"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"

TEST(TVLQR_Tests, DoubleIntegrator) {
  /////////////////////////////////////////////
  // Problem definition
  /////////////////////////////////////////////
  using Matrix = Eigen::Matrix<lqr_float, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<lqr_float, Eigen::Dynamic, 1>;

  // Set dimensions
  constexpr int N = 10;
  constexpr int dim = 2;
  float h = 0.01;

  int nx[N + 1];
  int nu[N];
  int n = 2 * dim;
  int m = dim;
  for (int k = 0; k < N; ++k) {
    nx[k] = n;
    nu[k] = m;
  }
  nx[N] = n;

  // Equilibrium
  Vector x0(n);
  Vector xeq(n);
  Vector ueq(m);
  xeq << 1, 2, 0, 0;
  ueq << 0, 0;

  // Initial state
  x0 = Vector::Zero(n);
  x0 << 10.5, -20.5, -4, 5;

  // Calculate A,B,f
  Matrix jac(n, n + m);
  Vector fk = Vector::Zero(n);

  discrete_double_integrator_dynamics(fk.data(), xeq.data(), ueq.data(), h, dim);
  discrete_double_integrator_jacobian(jac.data(), xeq.data(), ueq.data(), h, dim);
  Matrix Ak = jac.leftCols(n);
  Matrix Bk = jac.rightCols(m);

  // Cost
  Vector Qd = Vector::Constant(n, 1.1);
  Vector Rd = Vector::Constant(m, 0.1);
  Vector Qfd = Qd * 100;
  Matrix Qk = Qd.asDiagonal();
  Matrix Rk = Rd.asDiagonal();
  Matrix Qf = Qfd.asDiagonal();
  Vector qk = Vector::Constant(n, 0.01);
  Vector rk = Vector::Constant(m, 0.001);

  /////////////////////////////////////////////
  // Memory initialization
  /////////////////////////////////////////////
  bool is_diag = true;
  int mem_size = tvlqr_TotalMemSize(nx, nu, N, is_diag);
  fmt::print("mem size = {}\n", mem_size);
  fmt::print("number of floats: {}\n", mem_size / sizeof(lqr_float));

  // Allocate memory
  auto *mem0 = (lqr_float *)malloc(mem_size);
  lqr_float *A[N];
  lqr_float *B[N];
  lqr_float *f[N];

  lqr_float *Q[N + 1];
  lqr_float *R[N];
  lqr_float *H[N];
  lqr_float *q[N + 1];
  lqr_float *r[N];

  lqr_float *K[N];
  lqr_float *d[N];

  lqr_float *P[N + 1];
  lqr_float *p[N + 1];

  lqr_float *Qxx[N];
  lqr_float *Quu[N];
  lqr_float *Qux[N];
  lqr_float *Qx[N];
  lqr_float *Qu[N];
  lqr_float *delta_V;

  lqr_float *Qxx_tmp[N];
  lqr_float *Quu_tmp[N];
  lqr_float *Qux_tmp[N];
  lqr_float *Qx_tmp[N];
  lqr_float *Qu_tmp[N];

  lqr_float *x[N+1];
  lqr_float *u[N];
  lqr_float *y[N+1];

  lqr_float *mem = mem0;
  for (int k = 0; k < N; ++k) {
    // clang-format off
    // Assign pointers to allocated memory
    x[k] = mem; mem += n;
    u[k] = mem; mem += m;
    y[k] = mem; mem += n;

    A[k] = mem; mem += n * n;
    B[k] = mem; mem += n * m;
    f[k] = mem; mem += n;

    Q[k] = mem; mem += n;
    q[k] = mem; mem += n;
    R[k] = mem; mem += m;
    r[k] = mem; mem += m;

    K[k] = mem; mem += m * n;
    d[k] = mem; mem += m;

    P[k] = mem; mem += n * n;
    p[k] = mem; mem += n;

    Qxx[k] = mem; mem += n * n;
    Quu[k] = mem; mem += m * m;
    Qux[k] = mem; mem += m * n;
    Qx[k] = mem; mem += n;
    Qu[k] = mem; mem += m;

    Qxx_tmp[k] = mem; mem += n * n;
    Quu_tmp[k] = mem; mem += m * m;
    Qux_tmp[k] = mem; mem += m * n;
    Qx_tmp[k] = mem; mem += n;
    Qu_tmp[k] = mem; mem += m;

    // Assign problem data
    Eigen::Map<Matrix>(A[k], n, n) = Ak;
    Eigen::Map<Matrix>(B[k], n, m) = Bk;
    Eigen::Map<Vector>(f[k], n) = fk;

    Eigen::Map<Vector>(Q[k], n) = Qd;
    Eigen::Map<Vector>(R[k], m) = Rd;
    Eigen::Map<Vector>(q[k], n) = qk;
    Eigen::Map<Vector>(r[k], m) = rk;
  }
  // Terminal knot point
  x[N] = mem; mem += n;
  y[N] = mem; mem += n;
  Q[N] = mem; mem += n;
  q[N] = mem; mem += n;
  P[N] = mem; mem += n * n;
  p[N] = mem; mem += n;
  delta_V = mem; mem += 2;
  // clang-format on

  Eigen::Map<Vector>(Q[N], n) = Qfd;
  Eigen::Map<Vector>(q[N], n) = qk;
  EXPECT_EQ(mem - mem0, mem_size / sizeof(lqr_float));

  /////////////////////////////////////////////
  // Backward Pass
  /////////////////////////////////////////////
  bool linear_only_update = false;
  lqr_float reg = 0.0;
  Eigen::internal::set_is_malloc_allowed(false);
  // clang-format off
  int res = tvlqr_BackwardPass(nx, nu, N, A, B, f, Q, R, H, q, r, reg,
                     K, d, P, p, delta_V,
                     Qxx, Quu, Qux, Qx, Qu,
                     Qxx_tmp, Quu_tmp, Qux_tmp, Qx_tmp, Qu_tmp,
                     linear_only_update, is_diag);
  // clang-format on
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_EQ(res, TVLQR_SUCCESS);

  Matrix K0_expected(m, n);
  Vector d0_expected(m);
  // clang-format off
  K0_expected << 0.7753129718046554, 0.0, 5.840445640045901, 0.0,
                 0.0, 0.7753129718046554, 0.0, 5.840445640045901;
  d0_expected << -7.634078625343007, -15.256221385516275;
  // clang-format on
  double K_err = (Eigen::Map<Matrix>(K[0], m, n) - K0_expected).norm();
  double d_err = (Eigen::Map<Vector>(d[0], m) - d0_expected).norm();
  fmt::print("K_err = {}\n", K_err);
  fmt::print("d_err = {}\n", d_err);
  EXPECT_LT(K_err, 1e-6);
  EXPECT_LT(d_err, 1e-6);

  /////////////////////////////////////////////
  // Forward Pass
  /////////////////////////////////////////////
  res = tvlqr_ForwardPass(nx, nu, N, A, B, f, K, d, P, p, x0.data(), x, u, y);

  Vector xN(n);
  Vector yN(n);
  xN << 20.165445369740308, -0.13732391651279308, -2.3724421496097037, 2.3113121303468707;
  yN << 2218.2089906714345, -15.09563081640724, -260.9586364570674, 254.2543343381558;
  double x_err = (Eigen::Map<Vector>(x[N], n) - xN).lpNorm<Eigen::Infinity>();
  double y_err = (Eigen::Map<Vector>(y[N], n) - yN).lpNorm<Eigen::Infinity>();
  fmt::print("x_err = {}\n", x_err);
  fmt::print("y_err = {}\n", y_err);
  EXPECT_LT(x_err, 1e-6);
  EXPECT_LT(y_err, 1e-5);
}