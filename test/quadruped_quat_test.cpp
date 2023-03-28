//
// Created by Zixin Zhang on 3/27/23.
// Copyright (c) 2023 Robotic Exploration Lab. All rights reserved.
//

#include <chrono>
#include <filesystem>
#include <iostream>

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "altro/utils/quaternion_utils.hpp"
#include "fmt/chrono.h"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test_utils.hpp"

using Eigen::MatrixXd;

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace altro;

TEST(QuadrupedQuatTest, Dynamics) {
  int n = QuadrupedQuaternionModel::NumStates;
  int m = QuadrupedQuaternionModel::NumInputs;
  double h = 0.01;
  Vector x_dot(n);
  Vector x(n);
  Vector u(m);
  Eigen::Matrix<double, 3, 4> foot_pos_body;
  Eigen::Matrix<double, 3, 3> inertia_body;

  x << 0.1, 0.2, 0.3, 0.9180331, 0.073467, 0.0638159, 0.3843767, 0.1, 0.2, -0.1, 0.1, 0.2, 0.3;
  u << 0, 0, 13 * 9.81 / 4 - 5, 0, 0, 13 * 9.81 / 4 - 10, 0, 0, 13 * 9.81 / 4 + 10, 0, 0,
      13 * 9.81 / 4 + 15;
  foot_pos_body << 0.17, 0.17, -0.17, -0.17, 0.13, -0.13, 0.13, -0.13, -0.3, -0.3, -0.3, -0.3;
  inertia_body << 0.0158533, 0.0, 0.0, 0.0, 0.0377999, 0.0, 0.0, 0.0, 0.0456542;

  auto model_ptr = std::make_shared<QuadrupedQuaternionModel>();
  ContinuousDynamicsFunction ct_dyn = [model_ptr, foot_pos_body, inertia_body](
                                          double *x_dot, const double *x, const double *u) {
    model_ptr->Dynamics(x_dot, x, u, foot_pos_body, inertia_body);
  };
  ContinuousDynamicsJacobian ct_jac = [model_ptr, foot_pos_body, inertia_body](
                                          double *jac, const double *x, const double *u) {
    model_ptr->Jacobian(jac, x, u, foot_pos_body, inertia_body);
  };
  auto dt_dyn = MidpointDynamics(n, m, ct_dyn);
  auto dt_jac = MidpointJacobian(n, m, ct_dyn, ct_jac);

  // Continuous dynamics
  Vector x_dot_expected(n);
  x_dot_expected << 0.1000, 0.2000, -0.1000, -0.0677, 0.0170, 0.1000, 0.1419, 0, 0, 0.7692, -0.0297,
      179.9183, -0.0096;
  ct_dyn(x_dot.data(), x.data(), u.data());
  Vector x_dot_err = x_dot - x_dot_expected;
  for (int i = 0; i < n; i++) {
    EXPECT_LT(x_dot_err[i], 1e-4);
  }

  // Continuous Jacobian
  MatrixXd ct_jac_(n, n + m);
  MatrixXd ct_jac_expected(n, n + m);
  ct_jac_expected << 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0500, -0.1000,
      -0.1500, 0, 0, 0, -0.0367, -0.0319, -0.1922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0.0500, 0, 0.1500, -0.1000, 0, 0, 0, 0.4590, -0.1922, 0.0319, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0.1000, -0.1500, 0, 0.0500, 0, 0, 0, 0.1922, 0.4590, -0.0367, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0.1500, 0.1000, -0.0500, 0, 0, 0, 0, -0.0319, 0.0367, 0.4590, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0769, 0, 0, 0.0769, 0, 0,
      0.0769, 0, 0, 0.0769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0769, 0, 0, 0.0769, 0,
      0, 0.0769, 0, 0, 0.0769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0769, 0, 0, 0.0769,
      0, 0, 0.0769, 0, 0, 0.0769, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.1486, -0.0991, 0, 18.9235,
      8.2002, 0, 18.9235, -8.2002, 0, 18.9235, 8.2002, 0, 18.9235, -8.2002, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0.2365, 0, 0.0788, -7.9365, 0, -4.4974, -7.9365, 0, -4.4974, -7.9365, 0, 4.4974,
      -7.9365, 0, 4.4974, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0961, -0.0481, 0, -2.8475, 3.7236, 0,
      2.8475, 3.7236, 0, -2.8475, -3.7236, 0, 2.8475, -3.7236, 0;
  ct_jac(ct_jac_.data(), x.data(), u.data());
  MatrixXd ct_jac_err = ct_jac_ - ct_jac_expected;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n + m; j++) {
      EXPECT_LT(ct_jac_err(i, j), 1e-4);
    }
  }

  // Midpoint discrete dynamics
  Vector x_next = Vector::Zero(n);
  Vector x_next_expected = Vector::Zero(n);
  dt_dyn(x_next.data(), x.data(), u.data(), h);
  x_next_expected << 0.1010, 0.2020, 0.2990, 0.9171, 0.0719, 0.0689, 0.3861, 0.1000, 0.2000,
      -0.0923, 0.0984, 1.9992, 0.2995;
  Vector x_next_err = x_next - x_next_expected;
  for (int i = 0; i < n; i++) {
    EXPECT_LT(x_next_err[i], 1e-4);
  }

  // Midpoint discrete dynamics Jacobian
  MatrixXd dt_jac_(n, n + m);
  MatrixXd dt_jac_expected(n, n + m);
  dt_jac_expected << 1.0000, 0, 0, 0, 0, 0, 0, 0.0100, 0, 0, 0, 0, 0, 0.0000, 0, 0, 0.0000, 0, 0,
      0.0000, 0, 0, 0.0000, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0.0100, 0, 0, 0, 0, 0, 0.0000, 0, 0,
      0.0000, 0, 0, 0.0000, 0, 0, 0.0000, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0, 0.0100, 0, 0, 0, 0, 0,
      0.0000, 0, 0, 0.0000, 0, 0, 0.0000, 0, 0, 0.0000, 0, 0, 0, 1.0000, -0.0005, -0.0055, -0.0015,
      0, 0, 0, -0.0004, -0.0003, -0.0019, 0.0000, -0.0001, -0.0000, -0.0000, -0.0001, 0.0000,
      0.0000, 0.0000, -0.0000, -0.0000, 0.0000, 0.0000, 0, 0, 0, 0.0005, 1.0000, 0.0015, -0.0055, 0,
      0, 0, 0.0046, -0.0019, 0.0003, 0.0001, 0.0004, 0.0002, 0.0001, 0.0004, -0.0001, 0.0001,
      0.0004, 0.0001, 0.0001, 0.0004, -0.0002, 0, 0, 0, 0.0055, -0.0015, 1.0000, 0.0005, 0, 0, 0,
      0.0019, 0.0046, -0.0004, -0.0002, 0.0002, -0.0000, -0.0002, 0.0002, -0.0002, -0.0002, 0.0002,
      0.0002, -0.0002, 0.0002, 0.0000, 0, 0, 0, 0.0015, 0.0055, -0.0005, 1.0000, 0, 0, 0, -0.0003,
      0.0004, 0.0046, -0.0001, 0.0001, -0.0000, 0.0001, 0.0001, 0.0000, -0.0001, -0.0001, -0.0000,
      0.0001, -0.0001, 0.0000, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0.0008, 0, 0, 0.0008, 0,
      0, 0.0008, 0, 0, 0.0008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0, 0.0008, 0, 0,
      0.0008, 0, 0, 0.0008, 0, 0, 0.0008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000, 0, 0, 0, 0, 0,
      0.0008, 0, 0, 0.0008, 0, 0, 0.0008, 0, 0, 0.0008, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0000,
      -0.0015, -0.0054, 0.0001, 0.1891, 0.0820, -0.0000, 0.1891, -0.0820, 0.0001, 0.1893, 0.0820,
      -0.0000, 0.1893, -0.0820, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0024, 1.0000, 0.0008, -0.0794,
      0.0002, -0.0449, -0.0794, 0.0002, -0.0451, -0.0794, 0.0002, 0.0451, -0.0794, 0.0002, 0.0449,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0053, -0.0005, 1.0000, -0.0285, 0.0367, -0.0002, 0.0285,
      0.0367, 0.0002, -0.0285, -0.0377, -0.0002, 0.0285, -0.0377, 0.0002;
  dt_jac(dt_jac_.data(), x.data(), u.data(), h);
  MatrixXd dt_jac_err = dt_jac_ - dt_jac_expected;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n + m; j++) {
      EXPECT_LT(dt_jac_err(i, j), 1e-4);
    }
  }
}

TEST(QuadrupedQuatTest, MPC) {
  auto t_start = std::chrono::high_resolution_clock::now();
  const int n = QuadrupedQuaternionModel::NumStates;
  const int en = QuadrupedQuaternionModel::NumErrorStates;
  const int m = QuadrupedQuaternionModel::NumInputs;
  const int em = QuadrupedQuaternionModel::NumErrorInputs;
  const int N = 30;
  const double h = 0.01;
  Eigen::Matrix<double, 3, 4> foot_pos_body;
  Eigen::Matrix<double, 3, 3> inertia_body;
  foot_pos_body << 0.17, 0.17, -0.17, -0.17, 0.13, -0.13, 0.13, -0.13, -0.3, -0.3, -0.3, -0.3;
  inertia_body << 0.0158533, 0.0, 0.0, 0.0, 0.0377999, 0.0, 0.0, 0.0, 0.0456542;
  ALTROSolver solver(N);

  /// REFERENCES ///
//  Eigen::Vector3d r_start;
//  Eigen::Vector4d q_start;
//  Eigen::Vector3d v_start;
//  Eigen::Vector3d w_start;
//  r_start << 0.0, 0.0, 0.3;
//  q_start << 1.0, 0.0, 0.0, 0.0;
//  v_start << 0.0, 0.0, 0.0;
//  w_start << 0.0, 0.0, 0.0;
//
//  Eigen::Vector3d r_dot;
//  Eigen::Vector4d q_dot;
//  Eigen::Vector3d v_dot;
//  Eigen::Vector3d w_dot;
//  q_dot = 0.5 * altro::G(q_start) * w_start;
//  v_dot << 0.0, 0.0, 0.0;  // linear acceleration
//  w_dot << 0.3, 0.4, 0.5;  // angular acceleration
//
//  Eigen::Vector3d r_ref;
//  Eigen::Vector4d q_ref;
//  Eigen::Vector3d v_ref;
//  Eigen::Vector3d w_ref;
//  q_ref = q_start;
//  v_ref = v_start;
//  w_ref = w_start;
//
//  std::vector<Eigen::VectorXd> X_ref;
//  std::vector<Eigen::VectorXd> U_ref;
//  for (int i = 0; i <= N; ++i) {
//    Vector x_ref = Vector::Zero(n);
//    Vector u_ref = Vector::Zero(m);
//    double t = h * i;
//
//    r_ref = r_start + t * v_start + 0.5 * t * t * v_dot;
//    q_ref = q_ref + q_dot * h;
//    v_ref = v_ref + v_dot * h;
//    w_ref = w_ref + w_dot * h;
//
//    x_ref.head(3) = r_ref;
//    x_ref.segment<4>(3) = q_ref;
//    x_ref.segment<3>(7) = v_ref;
//    x_ref.tail(3) = w_ref;
//
//    X_ref.emplace_back(x_ref);
//    U_ref.emplace_back(u_ref);
//
//    q_dot = 0.5 * altro::G(q_ref) * w_ref;  // update q_dot
//  }

  // Try some simple reference
  std::vector<Eigen::VectorXd> X_ref;
  std::vector<Eigen::VectorXd> U_ref;

  for (int i = 0; i <= N; i++) {
    Vector x_ref = Vector::Zero(n);
    Vector u_ref = Vector::Zero(m);

    x_ref << 0.0, 0.0, 0.2,
             1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.15,
             0.0, 0.0, 0.0;
    u_ref << 0.0, 0.0, 13 * 9.81 / 4,
             0.0, 0.0, 13 * 9.81 / 4,
             0.0, 0.0, 13 * 9.81 / 4,
             0.0, 0.0, 13 * 9.81 / 4;

    X_ref.emplace_back(x_ref);
    U_ref.emplace_back(u_ref);
  }

  /// OBJECTIVE ///
  Eigen::Matrix<double, n, 1> Qd;
  Eigen::Matrix<double, m, 1> Rd;
//  double w = 1.0;
//
//  Qd << 1.0, 1.0, 1.0,
//        0, 0, 0, 0,
//        10.0, 10.0, 10.0,
//        10.0, 10.0, 10.0;

  double w = 3.0;
  Qd << 0.0, 0.0, 3.0,      // only track z position
        0.0, 0.0, 0.0, 0.0,  // ignore quaternion in Q
        0.1, 0.1, 0.1,       // track linear velocity
        0.1, 0.1, 3.0;       // track angular velocity

  Rd << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;

  /// DYNAMICS ///
  auto model_ptr = std::make_shared<QuadrupedQuaternionModel>();
  ContinuousDynamicsFunction ct_dyn = [model_ptr, foot_pos_body, inertia_body](
                                          double *x_dot, const double *x, const double *u) {
    model_ptr->Dynamics(x_dot, x, u, foot_pos_body, inertia_body);
  };
  ContinuousDynamicsJacobian ct_jac = [model_ptr, foot_pos_body, inertia_body](
                                          double *jac, const double *x, const double *u) {
    model_ptr->Jacobian(jac, x, u, foot_pos_body, inertia_body);
  };
  ExplicitDynamicsFunction dt_dyn = ForwardEulerDynamics(n, m, ct_dyn);
  ExplicitDynamicsJacobian dt_jac = ForwardEulerJacobian(n, m, ct_dyn, ct_jac);

  /// CONSTRAINTS ///
  float contacts[4] = {1.0, 1.0, 1.0, 1.0};  // FL, FR, RL, RR
  float mu = 0.7;
  float fz_max = 666;
  float fz_min = 5;
  auto friction_cone_con = [contacts, mu, fz_max, fz_min](a_float *c, const a_float *x,
                                                          const a_float *u) {
    (void)x;
    for (int i = 0; i < 4; i++) {
      c[0 + i * 6] = u[0 + i * 3] - mu * u[2 + i * 3];      // fx - mu*fz <= 0
      c[1 + i * 6] = -u[0 + i * 3] - mu * u[2 + i * 3];     // -fx - mu*fz <= 0
      c[2 + i * 6] = u[1 + i * 3] - mu * u[2 + i * 3];      // fy - mu*fz <= 0
      c[3 + i * 6] = -u[1 + i * 3] - mu * u[2 + i * 3];     // -fy - mu*fz <= 0
      c[4 + i * 6] = u[2 + i * 3] - fz_max * contacts[i];   // fz <= fz_max
      c[5 + i * 6] = -u[2 + i * 3] + fz_min * contacts[i];  // -fz + 5 <= 0
    }
  };

  auto friction_cone_jac = [mu](a_float *jac, const a_float *x, const a_float *u) {
    (void)x;
    (void)u;
    Eigen::Map<Eigen::Matrix<a_float, 24, 25>> J(jac);
    J.setZero();

    for (int i = 0; i < 4; i++) {
      J(0 + i * 6, 13 + i * 3) = 1;    // dc0/dfx
      J(0 + i * 6, 15 + i * 3) = -mu;  // dc0/dfz
      J(1 + i * 6, 13 + i * 3) = -1;   // dc1/dfx
      J(1 + i * 6, 15 + i * 3) = -mu;  // dc1/dfz
      J(2 + i * 6, 14 + i * 3) = 1;    // dc2/dfy
      J(2 + i * 6, 15 + i * 3) = -mu;  // dc2/dfz
      J(3 + i * 6, 14 + i * 3) = -1;   // dc3/dfy
      J(3 + i * 6, 15 + i * 3) = -mu;  // dc3/dfz
      J(4 + i * 6, 15 + i * 3) = 1;    // dc4/dfz
      J(5 + i * 6, 15 + i * 3) = -1;   // dc5/dfz
    }
  };

  /// SETUP ///
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 10;
  opts.use_backtracking_linesearch = true;
  opts.use_quaternion = true;
  opts.quat_start_index = 3;
  solver.SetOptions(opts);

  solver.SetDimension(n, m);
  solver.SetErrorDimension(en, em);
  solver.SetExplicitDynamics(dt_dyn, dt_jac);
  solver.SetTimeStep(h);

  // Cost function
  for (int i = 0; i <= N; i++) {
    solver.SetQuaternionCost(n, m, Qd.data(), Rd.data(), w, X_ref.at(i).data(), U_ref.at(i).data(),
                             i, 0);
  }

  solver.SetConstraint(friction_cone_con, friction_cone_jac, 24, ConstraintType::INEQUALITY,
                       "friction cone", 0, N + 1);

  Vector x_init = Vector::Zero(n);
  x_init << 0.0, 0.0, 0.0, // must be zero
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0;
  solver.SetInitialState(x_init.data(), n);
  solver.Initialize();

  // Initial guesses
  for (int i = 0; i <= N; i++) {
    solver.SetState(X_ref.at(i).data(), n, i);
  }
  solver.SetInput(U_ref.at(0).data(), m);

  /// SOLVE ///
  fmt::print("#############################################\n");
  fmt::print("                 MPC Solve\n");
  fmt::print("#############################################\n");

  solver.Solve();
  auto t_end = std::chrono::high_resolution_clock::now();
  using SecondsDouble = std::chrono::duration<double, std::ratio<1>>;
  SecondsDouble t_total = std::chrono::duration_cast<SecondsDouble>(t_end - t_start);
  fmt::print("Total time = {} ms\n", t_total * 1000);

  /// SAVE ///
  std::vector<Vector> X_sim;
  std::vector<Vector> U_sim;

  for (int k = 0; k <= N; k++) {
    Eigen::VectorXd x(n);
    solver.GetState(x.data(), k);
    X_sim.emplace_back(x);
    if (k != N) {
      Eigen::VectorXd u(m);
      solver.GetInput(u.data(), k);
      U_sim.emplace_back(u);
    }
  }

  // Save trajectory to JSON file
  std::filesystem::path out_file = "/home/zixin/Dev/ALTRO/test/quadruped_quat_test.json";
  std::ofstream traj_out(out_file);
  json X_ref_data(X_ref);
  json U_ref_data(U_ref);
  json X_data(X_sim);
  json U_data(U_sim);
  json data;
  data["reference_state"] = X_ref_data;
  data["reference_input"] = U_ref_data;
  data["state_trajectory"] = X_data;
  data["input_trajectory"] = U_data;
  traj_out << std::setw(4) << data;

  // Calculate tracking error
  double tracking_err = 0;
  for (int i = 0; i < N; i++) {
    tracking_err =
        tracking_err + (X_sim.at(i) - X_ref.at(i)).transpose() * (X_sim.at(i) - X_ref.at(i));
  }

  EXPECT_LT(tracking_err, 1e-4);
}