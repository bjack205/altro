//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "test_utils.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

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
  double dalpha_du = 1 / m;
  jac[0] = domega_dtheta;
  jac[1] = dalpha_dtheta;
  jac[2] = domega_domega;
  jac[3] = dalpha_domega;
  jac[4] = domega_du;
  jac[5] = dalpha_du;
}

altro::ExplicitDynamicsFunction MidpointDynamics(int n, int m, ContinuousDynamicsFunction f) {
  auto fd = [n, m, f](double *xn, const double *x, const double *u, float h) {
    static Eigen::VectorXd xm(n);
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

altro::ExplicitDynamicsJacobian MidpointJacobian(int n, int m, ContinuousDynamicsFunction f,
                                                 ContinuousDynamicsJacobian df) {
  auto fd = [n, m, f, df](double *jac, const double *x, const double *u, float h) {
    static Eigen::MatrixXd A(n, n);
    static Eigen::MatrixXd B(n, m);
    static Eigen::MatrixXd Am(n, n);
    static Eigen::MatrixXd Bm(n, m);
    static Eigen::VectorXd xm(n);
    static Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);

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

void BicycleModel::Dynamics(double *x_dot, const double *x, const double *u) const {
  double v = u[0];          // longitudinal velocity (m/s)
  double delta_dot = u[1];  // steering angle rage (rad/s)
  double theta = x[2];      // heading angle (rad) relative to x-axis
  double delta = x[3];      // steering angle (rad)

  double beta = 0;
  double omega = 0;
  double stheta = 0;
  double ctheta = 0;
  switch (reference_frame_) {
    case ReferenceFrame::CenterOfGravity:
      beta = std::atan2(distance_to_rear_wheels_ * delta, length_);
      omega = v * std::cos(beta) * std::tan(delta) / length_;
      stheta = std::sin(theta + beta);
      ctheta = std::cos(theta + beta);
      break;
    case ReferenceFrame::Rear:
      omega = v * tan(delta) / length_;
      stheta = std::sin(theta);
      ctheta = std::cos(theta);
      break;
    case ReferenceFrame::Front:
      omega = v * std::sin(delta) / length_;
      stheta = std::sin(theta + delta);
      ctheta = std::cos(theta + delta);
      break;
  };
  double px_dot = v * ctheta;
  double py_dot = v * stheta;
  x_dot[0] = px_dot;
  x_dot[1] = py_dot;
  x_dot[2] = omega;
  x_dot[3] = delta_dot;
}

void BicycleModel::Jacobian(double *jac, const double *x, const double *u) const {
  double v = u[0];          // longitudinal velocity (m/s)
  double theta = x[2];      // heading angle (rad) relative to x-axis
  double delta = x[3];      // steering angle (rad)

  Eigen::Map<Eigen::Matrix<double, 4, 6>> J(jac);
  double beta = 0;
  double dbeta_ddelta = 0;
  double by = 0;
  double bx = 0;
  double domega_ddelta = 0;
  double domega_dv = 0;

  double stheta = 0;
  double ctheta = 0;
  double ds_dtheta = 0;
  double dc_dtheta = 0;
  double ds_ddelta = 0;
  double dc_ddelta = 0;
  switch (reference_frame_) {
    case ReferenceFrame::CenterOfGravity:
      by = distance_to_rear_wheels_ * delta;
      bx = length_;
      beta = std::atan2(by, bx);
      dbeta_ddelta = bx / (bx * bx + by * by) * distance_to_rear_wheels_;
      domega_ddelta = v / length_ *
                      (-std::sin(beta) * std::tan(delta) * dbeta_ddelta +
                       std::cos(beta) / (std::cos(delta) * std::cos(delta)));
      domega_dv = std::cos(beta) * std::tan(delta) / length_;

      stheta = std::sin(theta + beta);
      ctheta = std::cos(theta + beta);
      ds_dtheta = +std::cos(theta + beta);
      dc_dtheta = -std::sin(theta + beta);
      ds_ddelta = +std::cos(theta + beta) * dbeta_ddelta;
      dc_ddelta = -std::sin(theta + beta) * dbeta_ddelta;
      break;
    case ReferenceFrame::Rear:
      domega_ddelta = v / length_ / (std::cos(delta) * std::cos(delta));
      domega_dv = std::tan(delta) / length_;

      stheta = std::sin(theta);
      ctheta = std::cos(theta);
      ds_dtheta = +std::cos(theta);
      dc_dtheta = -std::sin(theta);
      break;
    case ReferenceFrame::Front:
      domega_ddelta = v / length_ * std::cos(delta);
      domega_dv = std::sin(delta) / length_;

      stheta = std::sin(theta + delta);
      ctheta = std::cos(theta + delta);
      ds_dtheta = +std::cos(theta + delta);
      dc_dtheta = -std::sin(theta + delta);
      ds_ddelta = ds_dtheta;
      dc_ddelta = dc_dtheta;
      break;
  };
  J.setZero();
  J(0,2) = v * dc_dtheta;   // dxdot_dtheta
  J(0,3) = v * dc_ddelta;   // dxdot_ddelta
  J(0,4) = ctheta;          // dxdot_dv
  J(1,2) = v * ds_dtheta;   // dydot_dtheta
  J(1,3) = v * ds_ddelta;   // dydot_ddelta
  J(1,4) = stheta;          // dydot_dv
  J(2,3) = domega_ddelta;
  J(2,4) = domega_dv;
  J(3,5) = 1.0;
}

void ReadScottyTrajectory(int *Nref, float *tref, std::vector<Eigen::Vector4d> *xref,
                          std::vector<Eigen::Vector2d> *uref) {

  (void)Nref;
  (void)tref;
  (void)xref;
  (void)uref;
  const int n = 4;
  const int m = 2;
  fs::path test_dir = ALTRO_TEST_DIR;
  fs::path scotty_file = test_dir / "scotty.json";

  std::ifstream f(scotty_file);
  json data = json::parse(f);
  auto it = data.find("N");
  for (auto& el : data.items()) {
    if (el.key() == "N") {
      *Nref = el.value();
    }
    if (el.key() == "tf") {
      *tref = el.value();
    }
    if (el.key() == "state_trajectory") {
      int Nx = el.value().size();
      Eigen::Vector4d x;
      for (int k = 0; k < Nx; ++k) {
        json& x_json = el.value().at(k);
        if (x_json.size() != n) fmt::print("Got incorrect input size!\n");
        for (int i = 0; i < n; ++i) {
          x[i] = x_json[i];
        }
        xref->emplace_back(x);
      }
    }
    if (el.key() == "input_trajectory") {
      int Nu = el.value().size();
      Eigen::Vector2d u;
      for (int k = 0; k < Nu; ++k) {
        json& u_json = el.value().at(k);
        if (u_json.size() != m) fmt::print("Got incorrect input size!\n");
        for (int i = 0; i < m; ++i) {
          u[i] = u_json[i];
        }
        uref->emplace_back(u);
      }
    }
  }
  *Nref -= 1;
  f.close();
}

