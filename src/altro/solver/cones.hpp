//
// Created by Brian Jackson on 10/6/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "altro/solver/exceptions.hpp"
#include "altro/solver/typedefs.hpp"

namespace altro {

inline ConstraintType DualCone(ConstraintType cone) {
  ConstraintType dual_cone = ConstraintType::IDENTITY;
  switch (cone) {
    case ConstraintType::EQUALITY:
      dual_cone = ConstraintType::IDENTITY;
      break;
    case ConstraintType::INEQUALITY:
      dual_cone = ConstraintType::INEQUALITY;
      break;
    case ConstraintType::SECOND_ORDER_CONE:
      dual_cone = ConstraintType::SECOND_ORDER_CONE;
      break;
    case ConstraintType::IDENTITY:
      dual_cone = ConstraintType::EQUALITY;
      break;
  }
  return dual_cone;
}

inline bool ConicProjectionIsLinear(ConstraintType cone) {
  bool is_linear = true;
  switch (cone) {
    case ConstraintType::EQUALITY:
      is_linear = true;
      break;
    case ConstraintType::IDENTITY:
      is_linear = true;
      break;
    case ConstraintType::INEQUALITY:
      is_linear = true;
      break;
    case ConstraintType::SECOND_ORDER_CONE:
      is_linear = false;
      break;
  }
  return is_linear;
}

ErrorCodes ConicProjection(ConstraintType cone, int dim, const a_float *x, a_float *px);

ErrorCodes ConicProjectionJacobian(ConstraintType cone, int dim, const a_float *x, a_float* jac);

ErrorCodes ConicProjectionHessian(ConstraintType cone, int dim, const a_float *x, const a_float *b,
                                  a_float* hess);

}  // namespace altro