//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "test_utils.hpp"

#include <vector>

void discrete_double_integrator_dynamics(double *xnext, const double *x, const double *u, float h,
                                         int dim) {
  double b = h * h / 2;
  for (int i = 0; i < dim; ++i) {
    xnext[i] = x[i] + x[i + dim] * h + u[i] * b;
    xnext[i + dim] = x[i + dim] + u[i] * h;
  }
};

void discrete_double_integrator_jacobian(double *jac, const double *x, const double *u, float h,
                                         int dim) {
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
}

const double kPendulumMass = 1.0;
const double kPendulumLength = 0.5;
const double kPendulumFrictionCoeff = 0.1;
const double kPendulumInertia = 0.25;
const double kPendulumGravity = 9.81;

void pendulum_dynamics(double *xnext, const double *x, const double *u) {
  double l = kPendulumLength;
  double g = kPendulumGravity;
  double b = kPendulumFrictionCoeff;
  double m = kPendulumMass * l * l;

  double theta = x[0];
  double omega = x[1];

  double omega_dot = u[0] / m - g * std::sin(theta) / l - b * omega / m;
  xnext[0] = omega;
  xnext[1] = omega_dot;
}

void pendulum_jacobian(double *jac, const double *x, const double *u) {
  (void)u;
  double l = kPendulumLength;
  double g = kPendulumGravity;
  double b = kPendulumFrictionCoeff;
  double m = kPendulumMass * l * l;

  double domega_dtheta = 0.0;
  double domega_domega = 1.0;
  double domega_du = 0.0;
  double dalpha_dtheta = -g * std::cos(x[0]) / l;
  double dalpha_domega = -b / m;
  double dalpha_du = 1/m;
  jac[0] = domega_dtheta;
  jac[1] = dalpha_dtheta;
  jac[2] = domega_domega;
  jac[3] = dalpha_domega;
  jac[4] = domega_du;
  jac[5] = dalpha_du;
}
altro::ExplicitDynamicsFunction MidpointDynamics(int n, int m, ContinuousDynamicsFunction f) {
  auto fd = [n,m,f](double* xn, const double* x, const double* u, float h) {
    Eigen::VectorXd xm(n);
    Eigen::Map<Eigen::VectorXd> xn_vec(xn, n);
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<const Eigen::VectorXd> u_vec(u, n);
    f(xm.data(), x, u);
    xm *= h / 2;
    xm.noalias() += x_vec;
    f(xn, xm.data(), u);
    xn_vec = x_vec + h * xn_vec;
  };
  return fd;
}

altro::ExplicitDynamicsJacobian MidpointJacobian(int n, int m, ContinuousDynamicsFunction f, ContinuousDynamicsJacobian df) {
  auto fd = [n,m,f,df](double* jac, const double *x, const double *u, float h) {
    Eigen::MatrixXd A(n, n);
    Eigen::MatrixXd B(n, m);
    Eigen::MatrixXd Am(n, n);
    Eigen::MatrixXd Bm(n, m);
    Eigen::VectorXd xm(n);
    Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);

    Eigen::Map<Eigen::MatrixXd> J(jac, n, n + m);
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<const Eigen::VectorXd> u_vec(u, n);

    // Evaluate the midpoint
    f(xm.data(), x, u);
    xm = x_vec + h / 2 * xm;

    // Evaluate the Jacobian
    df(J.data(), x, u);
    A = J.leftCols(n);
    B = J.rightCols(m);

    // Evaluate the Jacobian at the midpoint
    df(J.data(), xm.data(), u);
    Am = J.leftCols(n);
    Bm = J.rightCols(m);

    // Apply the chain rule
    J.leftCols(n) = In + h * Am * (In + h / 2 * A);
    J.rightCols(m) = h * (Am * h / 2 * B + Bm);
  };
  return fd;
}
