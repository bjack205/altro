//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <memory>

#include "internal_types.hpp"
#include "shifted_vector.hpp"
#include "knotpoint_data.hpp"
#include "solver_options.hpp"
#include "solver_stats.hpp"

#include "altro/augmented_lagrangian/al_solver.hpp"

namespace altro {

class KnotPointData;

class SolverImpl {
 public:
  SolverImpl(int N) : horizon_length_(N), nx_(N+1,0), nu_(N+1,0), h_(N), problem_(N), opts(), alsolver_(N)  {
    altro::TrajectoryXXd traj(0,0,N);
    trajectory_ = std::make_shared<altro::TrajectoryXXd>(traj);
  }

  bool IsInitialized() const { return is_initialized_; }
  bool Initialize();
  void Solve();

  // Problem definition
  int horizon_length_;
  std::vector<int> nx_;   // number of states
  std::vector<int> nu_;   // number of inputs
  std::vector<float> h_;  // time steps

  // Solver
  AltroOptions opts;
  AltroStats stats;

  // Old AltroCpp
  altro::problem::Problem problem_;
  altro::augmented_lagrangian::AugmentedLagrangianiLQR<Eigen::Dynamic, Eigen::Dynamic> alsolver_;
  std::shared_ptr<altro::TrajectoryXXd> trajectory_;

 private:
  void SetCppSolverOptions();

  bool is_initialized_ = false;

};

}
