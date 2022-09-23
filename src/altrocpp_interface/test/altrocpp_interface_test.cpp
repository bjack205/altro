//
// Created by Brian Jackson on 9/23/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altrocpp_interface/altrocpp_interface.hpp"

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/common/trajectory.hpp"
#include "altro/problem/problem.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "augmented_lagrangian/al_solver.hpp"
#include "examples/basic_constraints.hpp"
#include "fmt/chrono.h"
#include "fmt/core.h"
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

auto jac = [](double *jac, const double *x, const double *u, float h) -> void {
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

TEST(AltroCppInterfaceTest, DoubleIntegrator) {
  // Create Model
  const int n = 2 * dim;
  const int m = dim;
  cpp_interface::GeneralDiscreteDynamics model(n, m, dyn, jac);

  // Check model derivatives
  bool jac_check = model.CheckJacobian();
  bool hess_check = model.CheckHessian();
  fmt::print("Jacobian check: {}\n", jac_check);
  fmt::print("Hessian check: {}\n", hess_check);

  Eigen::VectorXd xnext(n);
  Eigen::VectorXd x(n);
  Eigen::VectorXd u(m);
  Eigen::MatrixXd J(n, n + m);
  x << 0.1, 0.2, 0.3, 0.4;
  u << 10.1, -20.4;
  float h = 0.01;
  model.Evaluate(x, u, 0.0, h, xnext);
  fmt::print("xnext = [{}]\n", xnext.transpose().eval());
  dyn(xnext.data(), x.data(), u.data(), h);
  fmt::print("xnext = [{}]\n", xnext.transpose().eval());

  // Discretization
  float tf = 5.0;
  int num_segments = 10;
  h = tf / static_cast<double>(num_segments);

  // Create solver
  ALTROSolver solver(num_segments);
  solver.SetDimension(n, m, 0, LastIndex);
  solver.SetTimeStep(h, 0, LastIndex);

  // Set up the problem
  problem::Problem &prob = solver.solver_->problem_;
  fmt::print("N = {}\n", prob.NumSegments());

  // Dynamics
  solver.SetExplicitDynamics(dyn, jac, 0, LastIndex);
  //  std::shared_ptr<cpp_interface::GeneralDiscreteDynamics> model_ptr =
  //      std::make_shared<cpp_interface::GeneralDiscreteDynamics>(model);
  //  for (int k = 0; k < num_segments; ++k) {
  //    prob.SetDynamics(model_ptr, k);
  //  }

  // Cost Function
  //  using altro::examples::QuadraticCost;
  using cpp_interface::QuadraticCost;
  Eigen::MatrixXd Qf = Eigen::VectorXd::Constant(n, 1.0).asDiagonal();
  Eigen::MatrixXd Q = Eigen::VectorXd::Constant(n, 1.0).asDiagonal();
  Eigen::MatrixXd R = Eigen::VectorXd::Constant(m, 1e-2).asDiagonal();
  Eigen::VectorXd xf = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd uf = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd x0(n);
  Eigen::VectorXd Qf_diag = Qf.diagonal();
  Eigen::VectorXd Q_diag = Q.diagonal();
  Eigen::VectorXd R_diag = R.diagonal();
  x0 << 1.0, 1.0, 0.0, 0.0;
  solver.SetLQRCost(Q_diag.data(), R_diag.data(), xf.data(), uf.data(), 0, num_segments);
  solver.SetLQRCost(Qf_diag.data(), R_diag.data(), xf.data(), uf.data(), num_segments);
  //  std::shared_ptr<QuadraticCost> stage_cost;
  //  std::shared_ptr<QuadraticCost> term_cost;
  //  stage_cost = std::make_shared<QuadraticCost>(QuadraticCost::LQRCost(Q, R, xf, uf));
  //  term_cost = std::make_shared<QuadraticCost>(QuadraticCost::LQRCost(Qf, R, xf, uf, true));
  //  for (int k = 0; k < num_segments; ++kSilent) {
  //    prob.SetCostFunction(stage_cost, k);
  //  }
  //  prob.SetCostFunction(term_cost, num_segments);

  // Initial state
  solver.SetInitialState(x0.data(), x0.size());
  //  prob.SetInitialState(x0);

  // Constraints
  auto goalcon = [n, xf](double *c, const double *x, const double *u) {
    (void)u;
    for (int i = 0; i < n; ++i) {
      c[i] = x[i] - xf[i];
    }
  };
  auto goaljac = [n, m](double *jac, const double *x, const double *u) {
    (void)x;
    (void)u;
    Eigen::Map<MatrixXd> J(jac, n, n + m);
    for (int i = 0; i < n; ++i) {
      J(i, i) = 1.0;
    }
  };
//  cpp_interface::EqualityConstraint goal_constraint(n, m, n, goalcon, goaljac, "Goal Constraint");
//  prob.SetConstraint(std::make_shared<cpp_interface::EqualityConstraint>(goal_constraint),
//                     num_segments);
  solver.SetConstraint(goalcon, goaljac, n, ConstraintType::EQUALITY, "Goal Constraint",
                       num_segments);

  // Initial Trajectory
  solver.Initialize();
  //  using KnotPointXXd = KnotPoint<Eigen::Dynamic, Eigen::Dynamic>;
  //  std::vector<KnotPointXXd> knotpoints;
  //  float t = 0.0;
  //  for (int k = 0; k < num_segments + 1; ++k) {
  //    VectorXd x(n);
  //    VectorXd u(m);
  //    x.setZero();
  //    u.setZero();
  //    KnotPointXXd z(x, u, t);
  //    knotpoints.push_back(std::move(z));
  //    t += h;  // note this results in roundoff error
  //  }
  //  std::shared_ptr<TrajectoryXXd> Z = std::make_shared<TrajectoryXXd>(knotpoints);
  //  std::shared_ptr<TrajectoryXXd> Z = solver.solver_->trajectory_;

  //  std::shared_ptr<TrajectoryXXd> Z = std::make_shared<TrajectoryXXd>(n, m, num_segments);
  //  Z->SetUniformStep(h);
  //  solver.solver_->trajectory_ = Z;
  solver.SetInput(uf.data(), uf.size(), 0, num_segments);
  std::shared_ptr<TrajectoryXXd> Z = solver.solver_->trajectory_;
  //  for (int k = 0; k < num_segments; ++k) {
  //    Z->Control(k) = uf;
  //  }
  //  Z->SetUniformStep(h);

  //  prob.SetConstraint(std::)
  //  num_segments);
  //  prob.SetConstraint(std::make_shared<examples::GoalConstraint>(xf), num_segments);

  // Build solver
  augmented_lagrangian::AugmentedLagrangianiLQR<Eigen::Dynamic, Eigen::Dynamic> alsolver(prob);
  alsolver.SetTrajectory(Z);
  fmt::print("Solver initialized!\n");

  // Set Options
  alsolver.GetOptions().constraint_tolerance = 1e-6;
  alsolver.GetOptions().verbose = altro::LogLevel::kSilent;
  alsolver.GetOptions().max_iterations_total = 200;
  alsolver.GetiLQRSolver().GetOptions().verbose = altro::LogLevel::kInner;

  // Solve
  using millisf = std::chrono::duration<double, std::milli>;
  auto start = std::chrono::high_resolution_clock::now();
  alsolver.Solve();
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<millisf>(stop - start);
  fmt::print("Total time: {}\n", duration);

  VectorXd x_0 = Z->State(0);
  VectorXd x_N = Z->State(num_segments);
  fmt::print("x0: \n[{}]\n", x_0.transpose().eval());
  fmt::print("xf: \n[{}]\n", x_N.transpose().eval());

  EXPECT_LT((x_N - xf).norm(), (x_0 - xf).norm());
  EXPECT_LT((x_N - xf).norm(), 1e-4);

  solver.GetState(x_0.data(), 0);
  solver.GetState(x_N.data(), num_segments);
  double dist_to_goal = (x_N - xf).norm();
  fmt::print("Distance to Goal: {}\n", dist_to_goal);
  EXPECT_LT(dist_to_goal, (x_0 - xf).norm());
}