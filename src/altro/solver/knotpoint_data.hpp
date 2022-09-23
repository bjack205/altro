//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "internal_types.hpp"

namespace altro {

class KnotPointData {
 public:
  KnotPointData() = default;

  void Instantiate();

  // General data
  int num_states;
  int num_inputs;

  // States and controls
  Vector x;  // state
  Vector u;  // input
  int h;     // time step

  Vector x_bar;  // temp state
  Vector u_bar;  // temp input

  // Dynamics
  ExplicitDynamicsFunction dynamics_function_;
  ExplicitDynamicsJacobian dynamics_jacobian_;
  Matrix dynamics_jac;
  Matrix dynamics_val_;
  Vector dynamics_dual_;

  // Bound constraint bounds
  Vector x_hi_;
  Vector x_lo_;
  Vector u_hi_;
  Vector u_lo_;

  // Bound constraint values
  Vector c_x_hi_;
  Vector c_x_lo_;
  Vector c_u_hi_;
  Vector c_u_lo_;

  // Bound constraint duals
  Vector v_x_hi_;
  Vector v_x_lo_;
  Vector v_u_hi_;
  Vector v_u_lo_;

  // Constraints
  std::vector<int> constraint_dims_;
  std::vector<ConstraintType> constraint_type_;
  std::vector<ConstraintFunction> constraint_function_;
  std::vector<ConstraintJacobian> constraint_jacobian_;
  std::vector<Matrix> constraint_jac_;
  std::vector<Vector> constraint_val_;
  std::vector<Vector> constraint_dual_;

 private:

};

}  // namespace altro