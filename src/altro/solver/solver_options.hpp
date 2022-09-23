//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <limits>

#include "typedefs.hpp"

namespace altro {

enum class Verbosity { Silent, Outer, Inner };

struct AltroOptions {
  AltroOptions() = default;
  double tol_cost = 1e-4;
  double tol_cost_intermediate = 1e-4;
  double tol_primal_feasibility = 1e-4;
  double tol_stationarity = 1e-4;
//  double tol_complimentarity = 1e-4;
//  double tol_dual_feasibility = 1e-4;

  double max_state_value;
  double max_input_value;

  double penalty_initial = 1.0;
  double penalty_scaling = 10.0;
  double penalty_max = 1e8;

  Verbosity verbose = Verbosity::Silent;
  double max_solve_time = std::numeric_limits<a_float>::infinity();
  bool throw_errors = true;
};

}
