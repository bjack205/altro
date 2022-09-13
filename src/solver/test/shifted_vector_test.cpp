//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "gtest/gtest.h"
#include "solver/shifted_vector.hpp"
#include "Eigen/Dense"

namespace altro {

TEST(ShiftedVector, CheckSize) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  EXPECT_EQ(vec.size(), len);
}

TEST(ShiftedVector, CheckSingleShift) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  vec.ShiftStart();
  EXPECT_EQ(vec[0], 1);
  EXPECT_EQ(vec[1], 2);
  EXPECT_EQ(vec[2], 3);
  EXPECT_EQ(vec[3], 4);
  EXPECT_EQ(vec[4], 0);
}

TEST(ShiftedVector, CheckNegativeShift) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  vec.ShiftStart(-2);
  EXPECT_EQ(vec[0], 3);
  EXPECT_EQ(vec[1], 4);
  EXPECT_EQ(vec[2], 0);
  EXPECT_EQ(vec[3], 1);
  EXPECT_EQ(vec[4], 2);
}

TEST(ShiftedVector, ShiftByLength) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  vec.ShiftStart(len);
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(vec[i], i);
  }
}

TEST(ShiftedVector, SetStart2) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  vec.SetStart(2);
  EXPECT_EQ(vec[0], 2);
  EXPECT_EQ(vec[1], 3);
  EXPECT_EQ(vec[2], 4);
  EXPECT_EQ(vec[3], 0);
  EXPECT_EQ(vec[4], 1);
}

TEST(ShiftedVector, SetNegativeStart) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  vec.SetStart(-3);
  EXPECT_EQ(vec[0], 2);
  EXPECT_EQ(vec[1], 3);
  EXPECT_EQ(vec[2], 4);
  EXPECT_EQ(vec[3], 0);
  EXPECT_EQ(vec[4], 1);
}

TEST(ShiftedVector, IncShiftOverLength) {
  const int len = 5;
  ShiftedVector<int> vec(len);
  for (int i = 0; i < len; ++i) {
    vec[i] = i;
  }
  for (int i = 0; i < len + 3; ++i) {
    vec.ShiftStart();
  }
  EXPECT_EQ(vec[0], 3);
  EXPECT_EQ(vec[1], 4);
  EXPECT_EQ(vec[2], 0);
  EXPECT_EQ(vec[3], 1);
  EXPECT_EQ(vec[4], 2);
}

TEST(ShiftedVector, FillEigen) {
  const int len = 4;
  int n = 6;
  ShiftedVector<Eigen::VectorXd> vec(len, Eigen::VectorXd::Constant(n,3));
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(vec[i].size(), 6);
    EXPECT_EQ(vec[i][0], 3);
    vec[i][0] = i;
    vec[i][1] = 2 * i;
  }
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(vec[i][0], i);
    EXPECT_EQ(vec[i][1], 2 * i);
  }
}

}