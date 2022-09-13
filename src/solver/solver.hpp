//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "internal_types.hpp"
#include "shifted_vector.hpp"
#include "knotpoint_data.hpp"

#include "altro/augmented_lagrangian/al_solver.hpp"

namespace altro {

class KnotPointData;

class SolverImpl {
 public:
  SolverImpl(int N) : horizon_length_(N), nx_(N+1), nu_(N+1), problem_(N), alsolver_(N) {}

  // Problem definition
  int horizon_length_;
  std::vector<int> nx_;  // number of states
  std::vector<int> nu_;  // numver of inputs

  // Old AltroCpp
  altro::problem::Problem problem_;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<Eigen::Dynamic, Eigen::Dynamic> alsolver_;

};

}
