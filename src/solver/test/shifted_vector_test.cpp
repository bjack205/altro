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

TEST(ShiftedVector, IteratorConstructor) {
  std::vector<float> std_vec;
  const int len = 5;
  for (int i = 0; i < len; ++i) {
    std_vec.push_back(i * std::sqrt(i + 2));
  }
  ShiftedVector<float> vec(std_vec.begin(), std_vec.end());
  int shift = 2;
  vec.ShiftStart(shift);
  for (int i = 0; i < len; ++i) {
    int j = (i + shift) % len;
    EXPECT_EQ(vec[i], std_vec[j]);
  }
}

TEST(ShiftedVector, ConstIteratorConstructor) {
  std::vector<float> std_vec;
  const int len = 5;
  for (int i = 0; i < len; ++i) {
    std_vec.push_back(i * std::sqrt(i + 2));
  }
  ShiftedVector<float> vec(std_vec.cbegin(), std_vec.cend());
  int shift = 2;
  vec.ShiftStart(shift);
  for (int i = 0; i < len; ++i) {
    int j = (i + shift) % len;
    EXPECT_EQ(vec[i], std_vec[j]);
  }
}

TEST(ShiftedVector, CreateFromVector) {
  std::vector<float> std_vec;
  const int len = 5;
  for (int i = 0; i < len; ++i) {
    std_vec.push_back(i * std::sqrt(i + 2));
  }
  ShiftedVector<float> vec(std_vec);
  int shift = len + 1;
  vec.ShiftStart(shift);
  for (int i = 0; i < len; ++i) {
    int j = (i + shift) % len;
    EXPECT_EQ(vec[i], std_vec[j]);
  }
}

TEST(ShiftedVector, Copy) {
  const int len = 5;
  ShiftedVector<int> v0(len, 1);
  ShiftedVector<int> v1(v0);
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(v0[i], v1[i]);
  }
}

TEST(ShiftedVector, Move) {
  const int len = 5;
  ShiftedVector<int> v0(len, 2);
  ShiftedVector<int> v1(std::move(v0));
  for (int i = 0; i < len; ++i) {
    EXPECT_EQ(v1[i], 2);
  }
}


}