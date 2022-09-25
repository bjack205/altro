//
// Created by Brian Jackson on 9/24/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altro.h"

#include "Eigen/Dense"
#include "fmt/core.h"

int AltroSum(int a, int b) { return a + b; }

void SumVectors(double *a, double *b, int len) {
  Eigen::Map<Eigen::VectorXd> avec(a, len);
  Eigen::Map<Eigen::VectorXd> bvec(b, len);
  avec += bvec;
  fmt::print("Hi there!\n");
}