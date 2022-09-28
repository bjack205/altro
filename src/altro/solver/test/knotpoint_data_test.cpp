//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "gtest/gtest.h"
#include "altro/solver/knotpoint_data.hpp"
#include "altro/solver/internal_types.hpp"
#include "test_utils.hpp"
#include "fmt/core.h"

namespace altro {

TEST(KnotPointDataTests, Init) {
  bool is_terminal = false;
  KnotPointData data(is_terminal);
  EXPECT_FALSE(data.IsTerminalKnotPoint());
  EXPECT_FALSE(data.IsInitialized());
}

TEST(KnotPointDataTests, SetDimension) {
  bool is_terminal = false;
  KnotPointData data(is_terminal);
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
  KnotPointData data(is_terminal);
  ErrorCodes res;

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::StateDimUnknown);
  res = data.SetDimension(n, m);
  EXPECT_EQ(res, ErrorCodes::NoError);

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::NextStateDimUnknown);
  data.SetNextStateDimension(n);

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::TimestepNotPositive);
  data.SetTimestep(h);

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::DynamicsFunNotSet);
  data.SetLinearDynamics(n, n, m, A.data(), B.data(), f.data());

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::CostFunNotSet);
  data.SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);


  res = data.Instantiate();
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

  // Instantiate the terminal knot point
  KnotPointData term(true);
  res = term.SetDimension(n, 0);
  EXPECT_EQ(res, ErrorCodes::NoError);
  res = term.Instantiate();
  EXPECT_EQ(res, ErrorCodes::CostFunNotSet);
  term.SetDiagonalCost(n, 0, Qd.data(), nullptr, q.data(), nullptr, c);
  res = term.Instantiate();
  EXPECT_EQ(res, ErrorCodes::NoError);

  // Calc terminal cost-to-go
  term.x_ = x;
  res = term.CalcCostExpansion(true);
  EXPECT_EQ(res, ErrorCodes::NoError);
  EXPECT_TRUE(term.lxx_.isApprox(Q));
  EXPECT_TRUE(term.lx_.isApprox(Q * x + q));

  res = term.CalcTerminalCostToGo();
  EXPECT_EQ(res, ErrorCodes::NoError);
  EXPECT_TRUE(term.P_.isApprox(Q));
  EXPECT_TRUE(term.p_.isApprox(Q * x + q));

  // Calc Action-Value expansion
  res = data.CalcActionValueExpansion(term);
  EXPECT_EQ(res, ErrorCodes::NoError);
  EXPECT_TRUE(data.Qxx_.isApprox(Q + A.transpose() * Q * A));
  EXPECT_TRUE(data.Quu_.isApprox(R + B.transpose() * Q * B));
  EXPECT_TRUE(data.Qux_.isApprox(B.transpose() * Q * A));
  EXPECT_TRUE(data.Qx_.isApprox(Q * x + q + A.transpose() * (Q * x + q)));
  EXPECT_TRUE(data.Qu_.isApprox(R * u + r + B.transpose() * (Q * x + q)));

  // Calc Gains
  res = data.CalcGains();
  EXPECT_EQ(res, ErrorCodes::NoError);
  EXPECT_TRUE(data.d_.isApprox(data.Quu_.ldlt().solve(-data.Qu_)));
  EXPECT_TRUE(data.K_.isApprox(data.Quu_.ldlt().solve(data.Qux_)));

  // Calc cost-to-go
  res = data.CalcCostToGo();
  Matrix& P = Q;
  Matrix P_ = Q + A.transpose() * P * A - A.transpose() * P * B * (R + B.transpose() * P * B).llt().solve(B.transpose() * P * A);
  EXPECT_TRUE(P_.isApprox(data.P_));

  PrintErrorCode(res);

}

}
