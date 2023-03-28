//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "test_utils.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
}

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

altro::ExplicitDynamicsFunction ForwardEulerDynamics(int n, int m,
                                                     const ContinuousDynamicsFunction f) {
  auto fd = [n, m, f](double *xn, const double *x, const double *u, float h) {
    Eigen::Map<Eigen::VectorXd> xn_vec(xn, n);
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<const Eigen::VectorXd> u_vec(u, n);
    f(xn, x, u);  // xn is actually x_dot here
    xn_vec = x_vec + h * xn_vec;
  };
  return fd;
}

altro::ExplicitDynamicsJacobian ForwardEulerJacobian(int n, int m,
                                                     const ContinuousDynamicsFunction f,
                                                     const ContinuousDynamicsJacobian df) {
  auto fd = [n, m, f, df](double *jac, const double *x, const double *u, float h) {
    Eigen::Map<Eigen::MatrixXd> J(jac, n, n + m);
    Eigen::Map<const Eigen::VectorXd> x_vec(x, n);
    Eigen::Map<const Eigen::VectorXd> u_vec(u, n);

    static Eigen::MatrixXd In = Eigen::MatrixXd::Identity(n, n);

    df(J.data(), x, u);
    J.leftCols(n) = In + h * J.leftCols(n);
    J.rightCols(m) = h * J.rightCols(m);
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
  double v = u[0];      // longitudinal velocity (m/s)
  double theta = x[2];  // heading angle (rad) relative to x-axis
  double delta = x[3];  // steering angle (rad)

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
  J(0, 2) = v * dc_dtheta;  // dxdot_dtheta
  J(0, 3) = v * dc_ddelta;  // dxdot_ddelta
  J(0, 4) = ctheta;         // dxdot_dv
  J(1, 2) = v * ds_dtheta;  // dydot_dtheta
  J(1, 3) = v * ds_ddelta;  // dydot_ddelta
  J(1, 4) = stheta;         // dydot_dv
  J(2, 3) = domega_ddelta;
  J(2, 4) = domega_dv;
  J(3, 5) = 1.0;
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
  for (auto &el : data.items()) {
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
        json &x_json = el.value().at(k);
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
        json &u_json = el.value().at(k);
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

void SimpleQuaternionModel::Dynamics(double *x_dot, const double *x, const double *u) const {
  Eigen::Map<Eigen::VectorXd> x_dot_vec(x_dot, 4);
  Eigen::Map<const Eigen::VectorXd> x_vec(x, 4);
  Eigen::Map<const Eigen::VectorXd> u_vec(u, 3);

  x_dot_vec = 0.5 * altro::G(x_vec) * u_vec;
}

void SimpleQuaternionModel::Jacobian(double *jac, const double *x, const double *u) const {
  Eigen::Map<const Eigen::VectorXd> x_vec(x, 4);
  Eigen::Map<const Eigen::VectorXd> u_vec(u, 3);

  double qs = x_vec[0];
  double qa = x_vec[1];
  double qb = x_vec[2];
  double qc = x_vec[3];

  double wx = u_vec[0];
  double wy = u_vec[1];
  double wz = u_vec[2];

  Eigen::Map<Eigen::Matrix<double, 4, 7>> J(jac);
  J.setZero();
  J << 0, -wx / 2, -wy / 2, -wz / 2, -qa / 2, -qb / 2, -qc / 2, wx / 2, 0, wz / 2, -wy / 2, qs / 2,
      -qc / 2, qb / 2, wy / 2, -wz / 2, 0, wx / 2, qc / 2, qs / 2, -qa / 2, wz / 2, wy / 2, -wx / 2,
      0, -qb / 2, qa / 2, qs / 2;
}

void QuadrupedQuaternionModel::Dynamics(double *x_dot, const double *x, const double *u,
                                        Eigen::Matrix<double, 3, 4> foot_pos_body,
                                        Eigen::Matrix3d inertia_body) const {
  Eigen::Map<Eigen::VectorXd> x_dot_vec(x_dot, 13);
  Eigen::Map<const Eigen::VectorXd> x_vec(x, 13);
  Eigen::Map<const Eigen::VectorXd> u_vec(u, 12);

  double robot_mass = 13;
  Eigen::Vector3d g_vec;
  g_vec << 0, 0, -9.81;

  Eigen::Vector3d moment_body;
  moment_body = altro::skew(foot_pos_body.block<3, 1>(0, 0)) * u_vec.segment<3>(0) +
                altro::skew(foot_pos_body.block<3, 1>(0, 1)) * u_vec.segment<3>(3) +
                altro::skew(foot_pos_body.block<3, 1>(0, 2)) * u_vec.segment<3>(6) +
                altro::skew(foot_pos_body.block<3, 1>(0, 3)) * u_vec.segment<3>(9);

  // change rate of position
  x_dot_vec.segment<3>(0) = x_vec.segment<3>(7);
  // change rate of quaternion
  x_dot_vec.segment<4>(3) = 0.5 * altro::G(x_vec.segment<4>(3)) * x_vec.segment<3>(10);
  // change rate of linear velocity
  x_dot_vec.segment<3>(7) =
      (u_vec.segment<3>(0) + u_vec.segment<3>(3) + u_vec.segment<3>(6) + u_vec.segment<3>(9)) /
          robot_mass +
      g_vec;
  // change rate of angular velocity
  x_dot_vec.segment<3>(10) =
      inertia_body.inverse() *
      (moment_body - x_vec.segment<3>(10).cross(inertia_body * x_vec.segment<3>(10)));
}

void QuadrupedQuaternionModel::Jacobian(double *jac, const double *x, const double *u,
                                        Eigen::Matrix<double, 3, 4> foot_pos_body,
                                        Eigen::Matrix3d inertia_body) const {
  (void)u;
  Eigen::Map<Eigen::Matrix<double, 13, 25>> J(jac);  // jac = [dfc_dx, dfc_du]
  J.setZero();

  Eigen::Map<const Eigen::VectorXd> x_vec(x, 13);

  double robot_mass = 13;

  // Calculate dfc_dx
  Eigen::MatrixXd dfc_dx(13, 13);
  dfc_dx.setZero();
  // dv/dv
  dfc_dx.block<3, 3>(0, 7) = Eigen::Matrix3d::Identity();
  // dqdot/dq
  dfc_dx.block<1, 3>(3, 4) = -0.5 * x_vec.segment<3>(10).transpose();
  dfc_dx.block<3, 1>(4, 3) = 0.5 * x_vec.segment<3>(10);
  dfc_dx.block<3, 3>(4, 4) = -0.5 * altro::skew(x_vec.segment<3>(10));
  // dqdot/domega
  dfc_dx(3, 10) = -0.5 * x_vec(4);  // -0.5qa
  dfc_dx(3, 11) = -0.5 * x_vec(5);  // -0.5qb
  dfc_dx(3, 12) = -0.5 * x_vec(6);  // -0.5qc
  dfc_dx(4, 10) = 0.5 * x_vec(3);   // 0.5qs
  dfc_dx(4, 11) = -0.5 * x_vec(6);  // -0.5qc
  dfc_dx(4, 12) = 0.5 * x_vec(5);   // 0.5qb
  dfc_dx(5, 10) = 0.5 * x_vec(6);   // 0.5qc
  dfc_dx(5, 11) = 0.5 * x_vec(3);   // 0.5qs
  dfc_dx(5, 12) = -0.5 * x_vec(4);  // -0.5qa
  dfc_dx(6, 10) = -0.5 * x_vec(5);  // -0.5qb
  dfc_dx(6, 11) = 0.5 * x_vec(4);   // 0.5qa
  dfc_dx(6, 12) = 0.5 * x_vec(3);   // 0.5qs
  // domegadot/domega
  dfc_dx.block<3, 3>(10, 10) =
      -inertia_body.inverse() * (altro::skew(x_vec.segment<3>(10)) * inertia_body -
                                  altro::skew(inertia_body * x_vec.segment<3>(10)));

  // Calculate dfc_du
  Eigen::MatrixXd dfc_du(13, 12);
  dfc_du.setZero();

  for (int i = 0; i < 4; ++i) {
    dfc_du.block<3, 3>(7, 3 * i) = (1 / robot_mass) * Eigen::Matrix3d::Identity();
    dfc_du.block<3, 3>(10, 3 * i) =
        inertia_body.inverse() * altro::skew(foot_pos_body.block<3, 1>(0, i));
  }

  // Get Jacobian
  J.block<13, 13>(0, 0) = dfc_dx;
  J.block<13, 12>(0, 13) = dfc_du;
}

Eigen::Vector4d Slerp(Eigen::Vector4d q1, Eigen::Vector4d q2, double t) {
  if (q1 == q2) {
    return q1;
  } else {
    double dot = q1.dot(q2);
    double theta = acos(dot);
    double sinTheta = sin(theta);
    double a = sin((1 - t) * theta) / sinTheta;
    double b = sin(t * theta) / sinTheta;

    return a * q1 + b * q2;
  }
}