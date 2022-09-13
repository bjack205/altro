//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altro_solver.hpp"

#include "solver/solver.hpp"

namespace altro {

altro::ALTROSolver::ALTROSolver() : solver_(std::make_unique<SolverImpl>()) {}
ALTROSolver::ALTROSolver(const ALTROSolver& other)
    : solver_(std::make_unique<SolverImpl>(*other.solver_)) {}
ALTROSolver::ALTROSolver(ALTROSolver&& other) = default;
ALTROSolver& ALTROSolver::operator=(const ALTROSolver& other) {
  *this->solver_ = *other.solver_;
  return *this;
}
ALTROSolver& ALTROSolver::operator=(ALTROSolver&& other) = default;
altro::ALTROSolver::~ALTROSolver() = default;

void ALTROSolver::SetHorizonLength(int horizon_length) {}

}  // namespace altro
