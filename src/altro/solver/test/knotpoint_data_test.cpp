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
  EXPECT_EQ(res, ErrorCodes::CostFunNotSet);
  data.SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::DynamicsFunNotSet);
  data.SetLinearDynamics(n, n, m, A.data(), B.data(), f.data());

  res = data.Instantiate();
  EXPECT_EQ(res, ErrorCodes::NoError);

  PrintErrorCode(res);
}

}
