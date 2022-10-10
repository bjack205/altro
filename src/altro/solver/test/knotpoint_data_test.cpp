//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altro/solver/knotpoint_data.hpp"

#include "altro/solver/cones.hpp"
#include "altro/solver/internal_types.hpp"
#include "fmt/core.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"

namespace altro {

TEST(KnotPointDataTests, Init) {
  bool is_terminal = false;
  KnotPointData data(0, is_terminal);
  EXPECT_FALSE(data.IsTerminalKnotPoint());
  EXPECT_FALSE(data.IsInitialized());
}

TEST(KnotPointDataTests, SetDimension) {
  bool is_terminal = false;
  KnotPointData data(0, is_terminal);
  const int n = 4;
  const int m = 2;

  EXPECT_EQ(data.GetStateDim(), 0);
  EXPECT_EQ(data.GetInputDim(), 0);
  ErrorCodes err = data.SetDimension(n, m);

  EXPECT_EQ(err, ErrorCodes::NoError);
  EXPECT_EQ(data.GetStateDim(), n);
  EXPECT_EQ(data.GetInputDim(), m);

  err = data.SetDimension(n, 0);
  EXPECT_EQ(err, ErrorCodes::InputDimUnknown);
  err = data.SetDimension(-1, m);
  EXPECT_EQ(err, ErrorCodes::StateDimUnknown);
}

TEST(KnotPointDataTests, CalcCostExpansion) {
  const int dim = 2;
  int n = 2 * dim;
  int m = dim;
  float h = 0.01;
  Vector x(n);
  Vector u(m);
  x << 0.1, 0.2, -0.3, -1.1;
  u << 10.1, -20.2;

  Matrix jac(n, n + m);
  discrete_double_integrator_jacobian(jac.data(), x.data(), u.data(), h, dim);
  Vector Qd = Vector::Constant(n, 1.1);
  Vector Rd = Vector::Constant(m, 0.1);
  Matrix Q = Qd.asDiagonal();
  Matrix R = Rd.asDiagonal();
  Vector q = Vector::Constant(n, 0.01);
  Vector r = Vector::Constant(m, 0.001);
  float c = 10.5;

  Matrix A = jac.leftCols(n);
  Matrix B = jac.rightCols(m);
  Vector f = Vector::Zero(n);

  bool is_terminal = false;
  KnotPointData data(0, is_terminal);
  ErrorCodes res;

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::StateDimUnknown);
  res = data.SetDimension(n, m);
  EXPECT_EQ(res, ErrorCodes::NoError);

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::NextStateDimUnknown);
  data.SetNextStateDimension(n);

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::TimestepNotPositive);
  data.SetTimestep(h);

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::DynamicsFunNotSet);
  data.SetLinearDynamics(n, n, m, A.data(), B.data(), f.data());

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::CostFunNotSet);
  data.SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);

  res = data.Initialize();
  EXPECT_EQ(res, ErrorCodes::NoError);

  EXPECT_EQ(data.K_.rows(), m);
  EXPECT_EQ(data.K_.cols(), n);
  EXPECT_TRUE(data.A_.isApprox(A));
  EXPECT_TRUE(data.B_.isApprox(B));
  EXPECT_TRUE(data.f_.isApprox(f));
  EXPECT_TRUE(data.lxx_.diagonal().isApprox(Qd));
  EXPECT_TRUE(data.luu_.diagonal().isApprox(Rd));

  data.x_ = x;
  data.u_ = u;
  data.CalcCostExpansion(true);
  EXPECT_TRUE(data.lxx_.diagonal().isApprox(Qd));
  EXPECT_TRUE(data.luu_.diagonal().isApprox(Rd));
  EXPECT_TRUE(data.lux_.isApproxToConstant(0.0));
  EXPECT_TRUE(data.lx_.isApprox(Qd.asDiagonal() * x + q));
  EXPECT_TRUE(data.lu_.isApprox(Rd.asDiagonal() * u + r));

  data.CalcDynamicsExpansion();
  EXPECT_TRUE(data.A_.isApprox(A));
  EXPECT_TRUE(data.B_.isApprox(B));

  // Initialize the terminal knot point
  KnotPointData term(1, true);
  res = term.SetDimension(n, m);
  EXPECT_EQ(res, ErrorCodes::NoError);
  res = term.Initialize();
  EXPECT_EQ(res, ErrorCodes::CostFunNotSet);
  term.SetDiagonalCost(n, m, Qd.data(), nullptr, q.data(), nullptr, c);
  res = term.Initialize();
  EXPECT_EQ(res, ErrorCodes::NoError);

  // Calc terminal cost-to-go
  term.x_ = x;
  res = term.CalcCostExpansion(true);
  EXPECT_EQ(res, ErrorCodes::NoError);
  EXPECT_TRUE(term.lxx_.isApprox(Q));
  EXPECT_TRUE(term.lx_.isApprox(Q * x + q));

  PrintErrorCode(res);
}

class KnotPointConstraintTest : public ::testing::Test {
 public:
  KnotPointConstraintTest() : N(10), data(0, false) {}

 protected:
  void SetUp() override {}

  void InitializeKnotPoint(ConstraintType cone) {
    // States and controls
    n = 3;
    m = 2;
    p = 3;
    h = 0.01;
    x = Vector(n);
    u = Vector(m);
    z = Vector(p);
    x << 2, 2, 2;
    u << 10, 10;
    z << -1, 4, 10.1;
    rho = 1.2;

    // Obstacles
    double r1 = 1.0;
    double r2 = 2.0;
    Vector c1(3);
    Vector c2(3);
    c1 << 1, 2, 3;
    c2 << 4, 4, 4;

    // Constraint
    con = [=](double *c, const double *x, const double *u) {
      Eigen::Map<const Vector> x_vec(x, n);
      c[0] = r1 * r1 - (x_vec - c1).squaredNorm();
      c[1] = r2 * r2 - (x_vec - c2).squaredNorm();
      c[2] = u[0] + u[1];
    };
    con_jac = [=](double *jac, const double *x, const double *u) {
      (void)u;
      Eigen::Map<const Vector> x_vec(x, n);
      Eigen::Map<Matrix> J(jac, p, n + m);
      J.setZero();
      J.block(0, 0, 1, 3) = -2 * (x_vec - c1).transpose();
      J.block(1, 0, 1, 3) = -2 * (x_vec - c2).transpose();
      J.bottomRightCorner(1, 2).setConstant(1.0);
    };

    // Set dimensions
    data.SetDimension(n, m);
    data.SetNextStateDimension(n);
    data.SetTimestep(h);

    // Set Cost
    Vector Qd = Vector::Constant(n, 1.0);
    Vector Rd = Vector::Constant(m, 1.0);
    Vector q = Vector::Zero(n);
    Vector r = Vector::Zero(m);
    data.SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), 0.0);

    // Add Constraint
    data.SetConstraint(con, con_jac, p, cone, "constraint");

    // Set Dynamics
    Matrix A = Matrix::Identity(n, n);
    Matrix B = Matrix::Identity(n, m);
    B.bottomRightCorner(1, m).setConstant(1.0);
    data.SetLinearDynamics(n, n, m, A.data(), B.data(), nullptr);

    // Initialize
    ErrorCodes err;
    err = data.Initialize();
    EXPECT_EQ(err, ErrorCodes::NoError);
    EXPECT_TRUE(data.IsInitialized());

    // Set state and control
    data.x_ = x;
    data.u_ = u;

    data.SetPenalty(rho);
    data.z_[0] = z;
  }

  int N;
  int n;
  int m;
  int p;
  float h;

  Vector x;
  Vector u;
  Vector z;
  double rho;
  KnotPointData data;

  ConstraintFunction con;
  ConstraintJacobian con_jac;
};

TEST_F(KnotPointConstraintTest, Inequality) {
  InitializeKnotPoint(ConstraintType::INEQUALITY);

  Vector c(p);
  Matrix J(p, n + m);
  Vector c_expected(p);
  Matrix J_expected(p, n + m);
  // clang-format off
  c_expected <<  -1, -8, 20.0;
  J_expected <<
     -2.0, -0.0, 2.0, 0.0, 0.0,
      4.0,  4.0, 4.0, 0.0, 0.0,
      0.0,  0.0, 0.0, 1.0, 1.0;
  // clang-format on
  con(c.data(), x.data(), u.data());
  con_jac(J.data(), x.data(), u.data());
  EXPECT_LT((c - c_expected).norm(), 1e-6);
  EXPECT_LT((J - J_expected).norm(), 1e-6);

  // Calculate constraints
  data.CalcConstraints();
  EXPECT_LT((data.constraint_val_[0] - c_expected).norm(), 1e-6);

  // Calc AL Cost
  a_float alcost = data.CalcConstraintCosts();
  Vector z_tilde = z - rho * c;
  for (int i = 0; i < p; ++i) {
    if (z_tilde[i] > 0) z_tilde[i] = 0.0;
  }
  EXPECT_NEAR(alcost, z_tilde.squaredNorm() / (2 * rho), 1e-10);

  // Calc Constraint Jacobians
  data.CalcConstraintJacobians();
  EXPECT_LT((data.constraint_jac_[0] - J_expected).norm(), 1e-6);

  // Calc AL Cost Gradient
  Vector lx_expected(n);
  Vector lu_expected(m);
  lx_expected.setZero();
  lu_expected << 13.9, 13.9;
  data.lx_.setZero();
  data.lu_.setZero();
  data.CalcConstraintCostGradients();
  EXPECT_LT((data.lx_ - lx_expected).norm(), 1e-10);
  EXPECT_LT((data.lu_ - lu_expected).norm(), 1e-10);

  // Calc AL Cost Hessian
  data.lxx_.setZero();
  data.luu_.setZero();
  data.lux_.setZero();
  data.CalcConstraintCostHessians();
  EXPECT_DOUBLE_EQ(data.proj_hess_[0].norm(), 0.0);
  EXPECT_DOUBLE_EQ(data.lxx_.norm(), 0.0);
  EXPECT_DOUBLE_EQ(data.lux_.norm(), 0.0);
  EXPECT_TRUE(data.luu_.isApproxToConstant(1.2));
}

TEST_F(KnotPointConstraintTest, Equality) {
  InitializeKnotPoint(ConstraintType::EQUALITY);

  Vector c(p);
  Matrix J(p, n + m);
  Vector c_expected(p);
  Matrix J_expected(p, n + m);
  // clang-format off
  c_expected <<  -1, -8, 20.0;
  J_expected <<
     -2.0, -0.0, 2.0, 0.0, 0.0,
      4.0,  4.0, 4.0, 0.0, 0.0,
      0.0,  0.0, 0.0, 1.0, 1.0;
  // clang-format on
  con(c.data(), x.data(), u.data());
  con_jac(J.data(), x.data(), u.data());
  EXPECT_LT((c - c_expected).norm(), 1e-6);
  EXPECT_LT((J - J_expected).norm(), 1e-6);

  // Calculate constraints
  data.CalcConstraints();
  EXPECT_LT((data.constraint_val_[0] - c_expected).norm(), 1e-6);

  // Calc AL Cost
  a_float alcost = data.CalcConstraintCosts();
  Vector z_tilde = z - rho * c;
  EXPECT_NEAR(alcost, z_tilde.squaredNorm() / (2 * rho), 1e-10);

  // Calc Constraint Jacobians
  data.CalcConstraintJacobians();
  EXPECT_LT((data.constraint_jac_[0] - J_expected).norm(), 1e-6);

  // Calc AL Cost Gradient
  Vector lx_expected(n);
  Vector lu_expected(m);
  lx_expected << -54, -54.4, -54.8;
  lu_expected << 13.9, 13.9;
  data.lx_.setZero();
  data.lu_.setZero();
  data.CalcConstraintCostGradients();
  EXPECT_LT((data.lx_ - lx_expected).norm(), 1e-10);
  EXPECT_LT((data.lu_ - lu_expected).norm(), 1e-10);

  // Calc AL Cost Hessian
  data.lxx_.setZero();
  data.luu_.setZero();
  data.lux_.setZero();
  Matrix lxx_expected(n, n);
  lxx_expected << 24.0, 19.2, 14.399999999999999, 19.2, 19.2, 19.2, 14.399999999999999, 19.2, 24.0;
  data.CalcConstraintCostHessians();
  EXPECT_DOUBLE_EQ(data.proj_hess_[0].norm(), 0.0);
  EXPECT_DOUBLE_EQ((data.lxx_ - lxx_expected).norm(), 0.0);
  EXPECT_DOUBLE_EQ(data.lux_.norm(), 0.0);
  EXPECT_TRUE(data.luu_.isApproxToConstant(1.2));
}

TEST_F(KnotPointConstraintTest, SecondOrderCone_OutofCone) {
  InitializeKnotPoint(ConstraintType::SECOND_ORDER_CONE);
  z << -1, 4, 30.0;
  data.z_[0] << z;
  EXPECT_FALSE(ConicProjectionIsLinear(DualCone(ConstraintType::SECOND_ORDER_CONE)));

  Vector c(p);
  Matrix J(p, n + m);
  Vector c_expected(p);
  Matrix J_expected(p, n + m);
  // clang-format off
  c_expected <<  -1, -8, 20.0;
  J_expected <<
             -2.0, -0.0, 2.0, 0.0, 0.0,
      4.0,  4.0, 4.0, 0.0, 0.0,
      0.0,  0.0, 0.0, 1.0, 1.0;
  // clang-format on
  con(c.data(), x.data(), u.data());
  con_jac(J.data(), x.data(), u.data());
  EXPECT_LT((c - c_expected).norm(), 1e-6);
  EXPECT_LT((J - J_expected).norm(), 1e-6);

  // Calculate constraints
  data.CalcConstraints();
  EXPECT_LT((data.constraint_val_[0] - c_expected).norm(), 1e-6);

  // Calc AL Cost
  a_float alcost = data.CalcConstraintCosts();
  a_float alcost_expected = 80.04534293850527;
  EXPECT_NEAR(alcost, alcost_expected, 1e-10);

  // Calc Constraint Jacobians
  data.CalcConstraintJacobians();
  EXPECT_LT((data.constraint_jac_[0] - J_expected).norm(), 1e-6);

  // Calc AL Cost Gradient
  Vector lx_expected(n);
  Vector lu_expected(m);
  lx_expected << -38.910476877919685, -39.19870263257094, -39.4869283872222;
  lu_expected << -9.800735254367721, -9.800735254367721;
  data.lx_.setZero();
  data.lu_.setZero();
  data.CalcConstraintCostGradients();
  EXPECT_LT((data.lx_ - lx_expected).norm(), 1e-10);
  EXPECT_LT((data.lu_ - lu_expected).norm(), 1e-10);

  // Calc AL Cost Hessian
  data.lxx_.setZero();
  data.luu_.setZero();
  data.lux_.setZero();
  Matrix hess_expected(n + m, n + m);
  hess_expected << 13.121659323998685, 9.632047409257103, 6.142435494515529, 2.3820953755839365,
      2.3820953755839365, 9.632047409257108, 9.600915640264486, 9.569783871271873,
      2.399740526514188, 2.399740526514188, 6.142435494515531, 9.569783871271868,
      12.997132248028219, 2.417385677444439, 2.417385677444439, 2.382095375583937,
      2.3997405265141882, 2.4173856774444396, 0.6, 0.6, 2.382095375583937, 2.3997405265141882,
      2.4173856774444396, 0.6, 0.6;
  data.CalcConstraintCostHessians();
  EXPECT_LT((data.constraint_hess_[0] - hess_expected).norm(), 1e-6);
}

TEST_F(KnotPointConstraintTest, SecondOrderCone_BelowCone) {
  InitializeKnotPoint(ConstraintType::SECOND_ORDER_CONE);

  Vector c(p);
  Matrix J(p, n + m);
  Vector c_expected(p);
  Matrix J_expected(p, n + m);
  // clang-format off
  c_expected <<  -1, -8, 20.0;
  J_expected <<
             -2.0, -0.0, 2.0, 0.0, 0.0,
      4.0,  4.0, 4.0, 0.0, 0.0,
      0.0,  0.0, 0.0, 1.0, 1.0;
  // clang-format on
  con(c.data(), x.data(), u.data());
  con_jac(J.data(), x.data(), u.data());
  EXPECT_LT((c - c_expected).norm(), 1e-6);
  EXPECT_LT((J - J_expected).norm(), 1e-6);

  // Calculate constraints
  data.CalcConstraints();
  EXPECT_LT((data.constraint_val_[0] - c_expected).norm(), 1e-6);

  // Check that it's below the cone
  Vector z_bar = z - rho * c;
  EXPECT_LT(z_bar.head(p - 1).norm(), -z_bar[p-1]);

  // Calc AL Cost
  a_float alcost = data.CalcConstraintCosts();
  a_float alcost_expected = 0;
  EXPECT_NEAR(alcost, alcost_expected, 1e-10);

  // Calc Constraint Jacobians
  data.CalcConstraintJacobians();
  EXPECT_LT((data.constraint_jac_[0] - J_expected).norm(), 1e-6);

  // Calc AL Cost Gradient
  Vector lx_expected(n);
  Vector lu_expected(m);
  lx_expected.setZero();
  lu_expected.setZero();
  data.lx_.setZero();
  data.lu_.setZero();
  data.CalcConstraintCostGradients();
  EXPECT_LT((data.lx_ - lx_expected).norm(), 1e-10);
  EXPECT_LT((data.lu_ - lu_expected).norm(), 1e-10);

  // Calc AL Cost Hessian
  data.lxx_.setZero();
  data.luu_.setZero();
  data.lux_.setZero();
  Matrix hess_expected(n + m, n + m);
  hess_expected.setZero();
  data.CalcConstraintCostHessians();
  EXPECT_LT((data.constraint_hess_[0] - hess_expected).norm(), 1e-6);
}

TEST_F(KnotPointConstraintTest, SecondOrderCone_InCone) {
  InitializeKnotPoint(ConstraintType::SECOND_ORDER_CONE);
  z << -1, 4, 100.0;
  data.z_[0] << z;

  Vector c(p);
  Matrix J(p, n + m);
  Vector c_expected(p);
  Matrix J_expected(p, n + m);
  // clang-format off
  c_expected <<  -1, -8, 20.0;
  J_expected <<
             -2.0, -0.0, 2.0, 0.0, 0.0,
      4.0,  4.0, 4.0, 0.0, 0.0,
      0.0,  0.0, 0.0, 1.0, 1.0;
  // clang-format on
  con(c.data(), x.data(), u.data());
  con_jac(J.data(), x.data(), u.data());
  EXPECT_LT((c - c_expected).norm(), 1e-6);
  EXPECT_LT((J - J_expected).norm(), 1e-6);

  // Calculate constraints
  data.CalcConstraints();
  EXPECT_LT((data.constraint_val_[0] - c_expected).norm(), 1e-6);

  // Check that it's in the cone
  Vector z_bar = z - rho * c;
  EXPECT_LT(z_bar.head(p - 1).norm(), z_bar[p-1]);

  // Calc AL Cost
  a_float alcost = data.CalcConstraintCosts();
  a_float alcost_expected = 2483.75;
  EXPECT_NEAR(alcost, alcost_expected, 1e-10);

  // Calc Constraint Jacobians
  data.CalcConstraintJacobians();
  EXPECT_LT((data.constraint_jac_[0] - J_expected).norm(), 1e-6);

  // Calc AL Cost Gradient
  Vector lx_expected(n);
  Vector lu_expected(m);
  lx_expected << -54, -54.4, -54.8;
  lu_expected << -76, -76;
  data.lx_.setZero();
  data.lu_.setZero();
  data.CalcConstraintCostGradients();
  EXPECT_LT((data.lx_ - lx_expected).norm(), 1e-10);
  EXPECT_LT((data.lu_ - lu_expected).norm(), 1e-10);

  // Calc AL Cost Hessian
  data.lxx_.setZero();
  data.luu_.setZero();
  data.lux_.setZero();
  Matrix lxx_expected(n, n);
  lxx_expected << 24.0, 19.2, 14.399999999999999, 19.2, 19.2, 19.2, 14.399999999999999, 19.2, 24.0;
  data.CalcConstraintCostHessians();
  EXPECT_DOUBLE_EQ(data.proj_hess_[0].norm(), 0.0);
  EXPECT_DOUBLE_EQ((data.lxx_ - lxx_expected).norm(), 0.0);
  EXPECT_DOUBLE_EQ(data.lux_.norm(), 0.0);
  EXPECT_TRUE(data.luu_.isApproxToConstant(1.2));
}

}  // namespace altro
