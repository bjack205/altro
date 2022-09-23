//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <functional>

namespace altro {

using a_float = double;

class ALTROSolver;

constexpr int LastIndex = -1;

enum class SolveStatus {
  Success,
  MaxIterations,
  MaxObjectiveExceeded,
  StateOutOfBounds,
  InputOutOfBounds
};

using CallbackFunction = std::function<void(const ALTROSolver*)>;

using ExplicitDynamicsFunction =
    std::function<void(double* xnext, const double* x, const double* u, float h)>;

using ExplicitDynamicsJacobian =
    std::function<void(double* jac, const double* x, const double* u, float h)>;

using ImplicitDynamicsFunction = std::function<void(double* err, const double* x1, const double* u1,
                                                    const double* x2, const double* u2, float h)>;

using ImplicitDynamicsJacobian =
    std::function<void(double* jac1, double* jac2, const double x1, const double* u1,
                       const double* x2, const double* u2, float h)>;

using CostFunction = std::function<a_float(const a_float* x, const a_float* u)>;

using CostGradient =
    std::function<void(a_float* dx, a_float* du, const a_float* x, const a_float* u)>;

using CostHessian = std::function<void(a_float* ddx, a_float* ddu, a_float* dxdu, const a_float* x,
                                       const a_float* u)>;

using ConstraintFunction = std::function<void(a_float* val, const a_float* x, const a_float* u)>;
using ConstraintJacobian = std::function<void(a_float* jac, const a_float* x, const a_float* u)>;

enum class ConstraintType { EQUALITY, INEQUALITY, SECOND_ORDER_CONE };

class ConstraintIndex {
 public:
  int KnotPointIndex() const { return k; }

  friend ALTROSolver;

 private:
  ConstraintIndex(int k, int i) : k(k), i(i) {}

  int k;
  int i;
};

}  // namespace altro
