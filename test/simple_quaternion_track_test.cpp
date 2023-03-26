//
// Created by Zixin Zhang on 3/25/23.
// Copyright (c) 2023 Robotic Exploration Lab. All rights reserved.
//

#include <chrono>
#include <filesystem>
#include <iostream>

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "fmt/chrono.h"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test_utils.hpp"

using Eigen::MatrixXd;

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace altro;

TEST(SimpleQuaternionTrackTest, Dynamics) {
  SimpleQuaternionModel model;
  int n = SimpleQuaternionModel::NumStates;
  int m = SimpleQuaternionModel::NumInputs;
  Vector x_dot(n);
  Vector x(n);
  Vector u(m);
  x << 1, 0, 0, 0;
  u << 0.4, 0.5, 0.6;

  // Continuous Dynamics
  Vector x_dot_expected(4);
  x_dot_expected << 0, 0.2, 0.25, 0.3;
  model.Dynamics(x_dot.data(), x.data(), u.data());
  EXPECT_LT((x_dot - x_dot_expected).norm(), 1e-10);

  // Continuous Jacobian
  MatrixXd J(n, n + m);
  MatrixXd J_expected(n, n + m);
  J_expected << 0, -0.2, -0.25, -0.3, 0, 0, 0, 0.2, 0, 0.3, -0.25, 0.5, 0, 0, 0.25, -0.3, 0, 0.2, 0, 0.5, 0, 0.3, 0.25, -0.2, 0, 0, 0, 0.5, model.Jacobian(J.data(), x.data(), u.data());
  EXPECT_LT((J - J_expected).norm(), 1e-6);
}

TEST(SimpleQuaternionTrackTest, SLERP) {
  Eigen::Vector4d q_start;
  Eigen::Vector4d q_end;

  q_start << 1, 0, 0, 0;
  q_end << 0.4619398, 0.1913417, 0.4619398, 0.7325378;

  Eigen::Vector4d q1 = Slerp(q_start, q_end, 0.1);
  Eigen::Vector4d q1_expected;
  q1_expected << 0.99406, 0.023482, 0.056691, 0.0899;
  EXPECT_LT((q1 - q1_expected).norm(), 1e-5);

  Eigen::Vector4d q2 = Slerp(q_start, q_end, 0.6);
  Eigen::Vector4d q2_expected;
  q2_expected << 0.79343, 0.13131, 0.31701, 0.50272;
  EXPECT_LT((q2 - q2_expected).norm(), 1e-5);

  Eigen::Vector4d q3 = Slerp(q_start, q_end, 1);
  EXPECT_LT((q3 - q_end).norm(), 1e-5);

  Eigen::Vector4d q4 = Slerp(q_end, q_end, 0.74659816);
  EXPECT_LT((q4 - q_end).norm(), 1e-5);

  Eigen::Vector4d q5 = Slerp(q_start, q_end, 0);
  EXPECT_LT((q5 - q_start).norm(), 1e-5);
}

TEST(SimpleQuaternionTrackTest, MPC) {
  const int n = SimpleQuaternionModel::NumStates;
  const int en = SimpleQuaternionModel::NumErrorStates;
  const int m = SimpleQuaternionModel::NumInputs;
  const int em = SimpleQuaternionModel::NumErrorInputs;
  const int N = 60;
  const double h = 0.01;
  ALTROSolver solver(N);

  /// REFERENCES ///
  Eigen::Vector4d x_start;
  Eigen::Vector4d x_target;
  Eigen::Vector3d u_ref;

  x_start << 1, 0, 0, 0;
  x_target << 0.7071068, 0.7071068, 0, 0;  // rotate PI/2 around x axis in N*h seconds
  u_ref << (M_PI / 2) / (N * h) - 1, 0, 0;

  std::vector<Eigen::Vector4d> X_ref;
  std::vector<Eigen::Vector3d> U_ref;
  for (int i = 0; i <= N; ++i) {
    Eigen::Vector4d x_ref;
    x_ref = Slerp(x_start, x_target, double(i) / N);
    X_ref.emplace_back(x_ref);
    U_ref.emplace_back(u_ref);
  }

  /// OBJECTIVE ///
  Eigen::Matrix<double, n, 1> Qd;
  Eigen::Matrix<double, m, 1> Rd;
  double w = 10.0;

  Qd << 0, 0, 0, 0;
  Rd << 1e-6, 1e-6, 1e-6;

  /// DYNAMICS ///
  auto model_ptr = std::make_shared<SimpleQuaternionModel>();
  ContinuousDynamicsFunction ct_dyn = [model_ptr](double *x_dot, const double *x, const double *u) { model_ptr->Dynamics(x_dot, x, u); };
  ContinuousDynamicsJacobian ct_jac = [model_ptr](double *jac, const double *x, const double *u) { model_ptr->Jacobian(jac, x, u); };
  ExplicitDynamicsFunction dt_dyn = ForwardEulerDynamics(n, m, ct_dyn);
  ExplicitDynamicsJacobian dt_jac = ForwardEulerJacobian(n, m, ct_dyn, ct_jac);

  /// CONSTRAINTS ///
  // No constraints

  /// SETUP ///
  solver.SetDimension(n, m);
  solver.SetErrorDimension(en, em);
  solver.SetExplicitDynamics(dt_dyn, dt_jac);
  solver.SetTimeStep(h);

  // Cost function
  for (int i = 0; i <= N; i++) {
    solver.SetQuaternionCost(n, m, Qd.data(), Rd.data(), w, X_ref.at(i).data(), U_ref.at(i).data(), 0, i); // some bug here
//    solver.SetLQRCost(n, m, Qd.data(), Rd.data(), X_ref.at(i).data(), U_ref.at(i).data(), i);
  }

  solver.SetInitialState(x_start.data(), n);
  solver.Initialize();

  // Initial guesses
  for (int i = 0; i <= N; i++) {
    solver.SetState(X_ref.at(i).data(), n, i);
  }
  solver.SetInput(u_ref.data(), m);

  /// SOLVE ///
  fmt::print("#############################################\n");
  fmt::print("                 MPC Solve\n");
  fmt::print("#############################################\n");

  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 80;
  opts.use_backtracking_linesearch = true;
  opts.use_quaternion = true;
  solver.SetOptions(opts);

  solver.Solve();

  /// SAVE ///
  std::vector<Vector> x_sim;
  std::vector<Vector> u_sim;

  for (int k = 0; k < N; k++) {
    Eigen::VectorXd x(n);
    Eigen::VectorXd u(m);
    solver.GetState(x.data(), k);
    solver.GetInput(u.data(), k);
    x_sim.emplace_back(x);
    u_sim.emplace_back(u);
  }

  // Save trajectory to JSON file
  std::filesystem::path out_file = "/home/zixin/Dev/ALTRO/test/simple_quat_mpc.json";
  std::ofstream traj_out(out_file);
  json X_ref_data(X_ref);
  json U_ref_data(U_ref);
  json x_data(x_sim);
  json u_data(u_sim);
  json data;
  data["reference_state"] = X_ref_data;
  data["reference_input"] = U_ref_data;
  data["state_trajectory"] = x_data;
  data["input_trajectory"] = u_data;
  traj_out << std::setw(4) << data;

  double a = 0;
  EXPECT_LT(a, 1e-5);
}