//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altro_solver.hpp"

#include "solver/solver.hpp"

namespace altro {

altro::ALTROSolver::ALTROSolver(int horizon_length)
    : solver_(std::make_unique<SolverImpl>(horizon_length)) {}
//ALTROSolver::ALTROSolver(const ALTROSolver& other)
//    : solver_(std::make_unique<SolverImpl>(*other.solver_)) {}
ALTROSolver::ALTROSolver(ALTROSolver&& other) = default;
//ALTROSolver& ALTROSolver::operator=(const ALTROSolver& other) {
//  *this->solver_ = *other.solver_;
//  return *this;
//}
ALTROSolver& ALTROSolver::operator=(ALTROSolver&& other) = default;
altro::ALTROSolver::~ALTROSolver() = default;

void ALTROSolver::SetDimension(int num_states, int num_inputs, int k_start, int k_stop) {
  for (int k = k_start; k < k_stop; ++k) {
    solver_->nx_[k] = num_states;
    solver_->nu_[k] = num_inputs;
  }
}

}  // namespace altro
