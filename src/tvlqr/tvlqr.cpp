//
// Created by Brian Jackson on 10/2/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

// #define EIGEN_RUNTIME_NO_MALLOC
#include "tvlqr.h"

#include <Eigen/Dense>

#include "altro/utils/formatting.hpp"

using ConstMatrix = Eigen::Map<const Eigen::Matrix<lqr_float, Eigen::Dynamic, Eigen::Dynamic>>;
using ConstVector = Eigen::Map<const Eigen::Vector<lqr_float, Eigen::Dynamic>>;
using Matrix = Eigen::Map<Eigen::Matrix<lqr_float, Eigen::Dynamic, Eigen::Dynamic>>;
using Vector = Eigen::Map<Eigen::Vector<lqr_float, Eigen::Dynamic>>;

int tvlqr_TotalMemSize(const int *nx, const int *nu, int num_horizon, bool is_diag) {
  if (!nx) return 0;
  if (!nu) return 0;
  int mem_size = 0;

  for (int k = 0; k <= num_horizon; ++k) {
    int n = nx[k];
    mem_size += is_diag ? n : n * n;  // Q
    mem_size += n;                    // q
    mem_size += n * n;                // P
    mem_size += n;                    // p
    mem_size += n;                    // x
    mem_size += n;                    // y

    if (k < num_horizon) {
      int m = nu[k];
      mem_size += n * n;  // A
      mem_size += n * m;  // B
      mem_size += n;      // f

      mem_size += is_diag ? m : m * m;  // R
      mem_size += is_diag ? 0 : m * n;  // H
      mem_size += m;                    // r

      mem_size += m * n;  // K
      mem_size += m;      // d

      mem_size += n * n;  // Qxx
      mem_size += m * m;  // Quu
      mem_size += m * n;  // Qux
      mem_size += n;      // Qx
      mem_size += m;      // Qu

      mem_size += n * n;  // Qxx_tmp
      mem_size += m * m;  // Quu_tmp
      mem_size += m * n;  // Qux_tmp
      mem_size += n;      // Qx_tmp
      mem_size += m;      // Qu_tmp

      mem_size += m;  // u
    }
  }
  mem_size += 2;  // delta_V

  return mem_size * (int)sizeof(lqr_float);
}

int tvlqr_BackwardPass(const int *nx, const int *nu, int num_horizon, const lqr_float *const *A,
                       const lqr_float *const *B, const lqr_float *const *f,
                       const lqr_float *const *Q, const lqr_float *const *R,
                       const lqr_float *const *H, const lqr_float *const *q,
                       const lqr_float *const *r, lqr_float reg, lqr_float **K, lqr_float **d,
                       lqr_float **P, lqr_float **p, lqr_float *delta_V, lqr_float **Qxx,
                       lqr_float **Quu, lqr_float **Qux, lqr_float **Qx, lqr_float **Qu,
                       lqr_float **Qxx_tmp, lqr_float **Quu_tmp, lqr_float **Qux_tmp,
                       lqr_float **Qx_tmp, lqr_float **Qu_tmp, bool linear_only_update,
                       bool is_diag) {
  int N = num_horizon;

  // TODO: Use this option
  (void) linear_only_update;

  // Terminal Cost-to-go
  Matrix P_N(P[N], nx[N], nx[N]);
  Vector p_N(p[N], nx[N]);
  delta_V[0] = 0;
  delta_V[1] = 0;
  if (is_diag) {
    P_N = ConstVector(Q[N], nx[N]).asDiagonal();
  } else {
    P_N = ConstMatrix(Q[N], nx[N], nx[N]);
  }
  p_N = ConstVector(q[N], nx[N]);

  for (int k = N - 1; k >= 0; --k) {
    int n = nx[k];
    int m = nu[k];
    Matrix P_next(P[k + 1], nx[k + 1], nx[k + 1]);
    Vector p_next(p[k + 1], nx[k + 1]);
//    fmt::print("P[{}]:\n{}\n", k, P_next.eval());
//    fmt::print("p[{}]: [{}]\n", k, p_next.transpose().eval());

    Matrix Qxx_k(Qxx[k], n, n);
    Matrix Quu_k(Quu[k], m, m);
    Matrix Qux_k(Qux[k], m, n);
    Vector Qx_k(Qx[k], n);
    Vector Qu_k(Qu[k], m);

    Matrix Qxx_(Qxx_tmp[k], n, n);
    Matrix Quu_(Quu_tmp[k], m, m);
    Matrix Qux_(Qux_tmp[k], m, n);
    Vector Qx_(Qx_tmp[k], n);
    Vector Qu_(Qu_tmp[k], m);

    ConstMatrix A_k(A[k], nx[k + 1], n);
    ConstMatrix B_k(B[k], nx[k + 1], m);
    ConstVector f_k(f[k], nx[k + 1]);

    ConstMatrix Q_k(Q[k], n, n);
    ConstMatrix R_k(R[k], m, m);
    ConstVector Qd_k(Q[k], n);
    ConstVector Rd_k(R[k], m);
    ConstMatrix H_k(H[k], m, n);
    ConstVector q_k(q[k], n);
    ConstVector r_k(r[k], m);

    // Action-value expansion
    if (is_diag) {
      Qxx_k = Qd_k.asDiagonal();
      Quu_k = Rd_k.asDiagonal();
      Qux_k.setZero();
    } else {
      Qxx_k = Q_k;
      Quu_k = R_k;
      Qux_k = H_k;
    }
    //  Qxx_k += A_k.transpose() * P_next * A_k;
    Qxx_.noalias() = A_k.transpose() * P_next;
    Qxx_k.noalias() += Qxx_ * A_k;

    // Quu_k += B_k.transpose() * P_next * B_k;
    Qux_.noalias() = B_k.transpose() * P_next;
    Quu_k.noalias() += Qux_ * B_k;

    // Qux_k += B_k.transpose() * P_next * A_k;
    Qux_k.noalias() += Qux_ * A_k;

    // Qx_k = q_k + A_k.transpose() * (P_next * f_k + p_next);
    // Qu_k = r_k + B_k.transpose() * (P_next * f_k + p_next);
    Qx_ = p_next;
    Qx_.noalias() += P_next * f_k;
    Qx_k = q_k;
    Qx_k.noalias() += A_k.transpose() * Qx_;
    Qu_k = r_k;
    Qu_k.noalias() += B_k.transpose() * Qx_;

    // Calc Gains
    Matrix K_k(K[k], m, n);
    Vector d_k(d[k], m);
    K_k = Qux_k;
    d_k = -Qu_k;
    Quu_ = Quu_k;
    Quu_.diagonal().array() += reg;
    Eigen::LLT<Eigen::Ref<Matrix>> Quu_fact(Quu_);
    if (Quu_fact.info() != Eigen::Success) {
      return k;
    }
    Quu_fact.solveInPlace(K_k);
    Quu_fact.solveInPlace(d_k);

    // Calculate cost-to-go
    Matrix P_k(P[k], n, n);
    Vector p_k(p[k], n);

    // P_k = Qxx_k + K_k.transpose() * Quu_k * K_k - K_k.transpose() * Qux_k - Qux_k.transpose() *
    P_k = Qxx_k;
    Qux_.noalias() = Quu_k * K_k;
    Qxx_.noalias() = K_k.transpose() * Qux_k;
    Qx_.noalias() = K_k.transpose() * Qu_k;
    P_k.noalias() += Qux_.transpose() * K_k;
    P_k.noalias() -= Qxx_;
    P_k.noalias() -= Qxx_.transpose();

    // K_k; p_k = Qx_k - K_k.transpose() * Quu_k * d_k - K_k.transpose() * Qu_k + Qux_k.transpose()
    // * d_k;
    p_k = Qx_k;
    p_k.noalias() -= Qux_.transpose() * d_k;
    p_k.noalias() -= K_k.transpose() * Qu_k;
    p_k.noalias() += Qux_k.transpose() * d_k;

    // Expected decrease in cost-to-go
    Qu_.noalias() = Quu_k * d_k;
    delta_V[0] += d_k.dot(Qu_k);
    delta_V[1] += 0.5 * d_k.dot(Qu_);
  }

  return TVLQR_SUCCESS;
}

int tvlqr_ForwardPass(const int *nx, const int *nu, int num_horizon, const lqr_float *const *A,
                      const lqr_float *const *B, const lqr_float *const *f,
                      const lqr_float *const *K, const lqr_float *const *d,
                      const lqr_float *const *P, const lqr_float *const *p, const lqr_float *x0,
                      lqr_float **x, lqr_float **u, lqr_float **y) {
  // Set Initial state
  Vector x_0(x[0], nx[0]);
  x_0 = ConstVector(x0, nx[0]);

  // Simulate forward the linear dynamics
  int N = num_horizon;
  for (int k = 0; k < N; ++k) {
    int n = nx[k];
    int m = nu[k];

    Vector x_k(x[k], n);
    Vector u_k(u[k], m);
    Vector x_n(x[k + 1], nx[k + 1]);

    ConstMatrix A_k(A[k], nx[k + 1], n);
    ConstMatrix B_k(B[k], nx[k + 1], m);
    ConstVector f_k(f[k], nx[k + 1]);

    ConstMatrix K_k(K[k], m, n);
    ConstVector d_k(d[k], m);

    u_k = d_k;
    u_k.noalias() -= K_k * x_k;

    x_n = f_k;
    x_n.noalias() += A_k * x_k;
    x_n.noalias() += B_k * u_k;

    if (y != nullptr) {
      ConstMatrix P_k(P[k], n, n);
      ConstVector p_k(p[k], n);
      Vector y_k(y[k], n);
      y_k = P_k * x_k + p_k;
    }
  }
  // Terminal knot point
  if (y != nullptr) {
    int k = N;
    int n = nx[N];
    ConstMatrix P_k(P[k], n, n);
    ConstVector p_k(p[k], n);
    Vector x_k(x[k], n);
    Vector y_k(y[k], n);
    y_k = P_k * x_k + p_k;
  }
  return TVLQR_SUCCESS;
}
