//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "gtest/gtest.h"
#include "fmt/core.h"

#include "altro/altro.hpp"

TEST(AltroAPI, Constructor) {
  fmt::print("Hello from testing suite!\n");

  altro::ALTROSolver solver;

  const int N = 10;
//  solver.SetHorizonLength(N);
}
