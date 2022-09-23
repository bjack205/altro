//
// Created by Brian Jackson on 9/23/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <chrono>

#include "typedefs.hpp"

namespace altro {

struct AltroStats {
  using millisd = std::chrono::duration<double, std::milli>;  // milliseconds in double

  SolveStatus status;
  millisd solve_time;
  int iterations;
  int outer_iterations;
  double objective_value;
  double stationarity;
  double primal_feasibility;
  double complimentarity;
};

}  // namespace altro
