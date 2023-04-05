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

//TEST(SimpleQuaternionTrackTest, EigenRemove) {
//  Vector vec_A;
//  Vector vec_A_1;
//  Vector vec_A_2;
//  Vector vec_A_3;
//
//  vec_A = Vector::Zero(5);
//  vec_A << 1, 2, 3, 4, 5;
//  std::cout << "vec_A = " << vec_A.transpose() << std::endl;
//
//  vec_A_1 = removeElement(vec_A, 4);
//  std::cout << "Remove the element at idx = 4, vec_A = " << vec_A_1.transpose() << std::endl;
//  EXPECT_LT(vec_A[0] - 1, 1e-5);
//  EXPECT_LT(vec_A[1] - 2, 1e-5);
//  EXPECT_LT(vec_A[2] - 3, 1e-5);
//  EXPECT_LT(vec_A[3] - 4, 1e-5);
//
//  vec_A_2 = removeElement(vec_A_1, 0);
//  std::cout << "Remove the element at idx = 0, vec_A = " << vec_A_2.transpose() << std::endl;
//  EXPECT_LT(vec_A[0] - 2, 1e-5);
//  EXPECT_LT(vec_A[1] - 3, 1e-5);
//  EXPECT_LT(vec_A[2] - 4, 1e-5);
//
//  vec_A_3 = removeElement(vec_A_2, 1);
//  std::cout << "Remove the element at idx = 1, vec_A = " << vec_A_3.transpose() << std::endl;
//  EXPECT_LT(vec_A[0] - 2, 1e-5);
//  EXPECT_LT(vec_A[1] - 4, 1e-5);
//}

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
  J_expected << 0, -0.2, -0.25, -0.3, 0, 0, 0, 0.2, 0, 0.3, -0.25, 0.5, 0, 0, 0.25, -0.3, 0, 0.2, 0,
      0.5, 0, 0.3, 0.25, -0.2, 0, 0, 0, 0.5, model.Jacobian(J.data(), x.data(), u.data());
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
  auto t_start = std::chrono::high_resolution_clock::now();
  const int n = SimpleQuaternionModel::NumStates;
  const int en = SimpleQuaternionModel::NumErrorStates;
  const int m = SimpleQuaternionModel::NumInputs;
  const int em = SimpleQuaternionModel::NumErrorInputs;
  const int N = 100;
  const double h = 0.01;
  ALTROSolver solver(N);

  /// REFERENCES ///
  Eigen::Vector4d x_start;
  Eigen::Vector4d x_target;
  Eigen::Vector3d u_ref;

  // rotate PI/6 around [0.2672612, 0.5345225, 0.8017837] in N*h seconds
  x_start << 1, 0, 0, 0;
  x_target << 0.9659258, 0.0691723, 0.1383446, 0.2075169;
  u_ref << 0.1399377 / (N * h), 0.2798753 / (N * h), 0.419813 / (N * h);

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
  ContinuousDynamicsFunction ct_dyn = [model_ptr](double *x_dot, const double *x, const double *u) {
    model_ptr->Dynamics(x_dot, x, u);
  };
  ContinuousDynamicsJacobian ct_jac = [model_ptr](double *jac, const double *x, const double *u) {
    model_ptr->Jacobian(jac, x, u);
  };
  ExplicitDynamicsFunction dt_dyn = ForwardEulerDynamics(n, m, ct_dyn);
  ExplicitDynamicsJacobian dt_jac = ForwardEulerJacobian(n, m, ct_dyn, ct_jac);

  /// CONSTRAINTS ///
  double omega_max = 0.5;
  auto upper_bound_con = [omega_max](a_float *c, const a_float *x, const a_float *u) {
    (void) x;
    c[0] = u[0] - omega_max;
    c[1] = u[1] - omega_max;
    c[2] = u[2] - omega_max;
  };

  auto upper_bound_jac = [](a_float *jac, const a_float *x, const a_float *u) {
    (void) x;
    (void) u;
    Eigen::Map<Eigen::Matrix<a_float, 3, n - 1 + m>> J(jac);
    J.setZero();
    J(0, 3) = 1;
    J(1, 4) = 1;
    J(2, 5) = 1;
  };

  /// SETUP ///
  AltroOptions opts;
  opts.verbose = Verbosity::Inner;
  opts.iterations_max = 80;
  opts.use_backtracking_linesearch = true;
  opts.use_quaternion = true;
  opts.quat_start_index = 0;
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

  solver.SetConstraint(upper_bound_con, upper_bound_jac, 3, ConstraintType::INEQUALITY,
                       "upper bound", 0, N);
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

  solver.Solve();
  auto t_end = std::chrono::high_resolution_clock::now();
  using SecondsDouble = std::chrono::duration<double, std::ratio<1>>;
  SecondsDouble t_total = std::chrono::duration_cast<SecondsDouble>(t_end - t_start);
  fmt::print("Total time = {} ms\n", t_total * 1000);

  /// SAVE ///
  std::vector<Vector> X_sim;
  std::vector<Vector> U_sim;

  for (int k = 0; k < N; k++) {
    Eigen::VectorXd x(n);
    Eigen::VectorXd u(m);
    solver.GetState(x.data(), k);
    solver.GetInput(u.data(), k);
    X_sim.emplace_back(x);
    U_sim.emplace_back(u);
  }

  // Save trajectory to JSON file
  fs::path test_dir = ALTRO_TEST_DIR;
  fs::path out_file = test_dir / "simple_quaternion_track_test.json";
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