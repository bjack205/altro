//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <limits>

#include "typedefs.hpp"

namespace altro {

struct SolverOptions {
  explicit SolverOptions(double max_solve_time);
  double tol_cost = 1e-4;
  double tol_cost_intermediate = 1e-4;
  double tol_primal_feasibility = 1e-4;
//  double tol_stationarity = 1e-4;
//  double tol_complimentarity = 1e-4;
//  double tol_dual_feasibility = 1e-4;

  double penalty_initial = 1.0;
  double penalty_scaling = 10.0;
  double penalty_max = 1e8;

  double max_solve_time = std::numeric_limits<a_float>::infinity();
  bool throw_errors = true;
};

}
