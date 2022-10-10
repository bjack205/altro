//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "exceptions.hpp"

#include "fmt/core.h"

namespace altro {

const char* ErrorCodeToString(ErrorCodes err) {
  switch (err) {
    case ErrorCodes::NoError:
      return "no error";
      break;
    case ErrorCodes::StateDimUnknown:
      return "state dimension unknown";
      break;
    case ErrorCodes::InputDimUnknown:
      return "input dimension unknown";
      break;
    case ErrorCodes::NextStateDimUnknown:
      return "next state dimension unknown";
      break;
    case ErrorCodes::DimensionUnknown:
      return "dimension unknown";
      break;
    case ErrorCodes::BadIndex:
      return "bad index";
      break;
    case ErrorCodes::DimensionMismatch:
      return "dimension mismatch";
      break;
    case ErrorCodes::SolverNotInitialized:
      return "solver not initialized";
      break;
    case ErrorCodes::SolverAlreadyInitialized:
      return "solver already initialized";
      break;
    case ErrorCodes::NonPositive:
      return "expected a positive value";
      break;
    case ErrorCodes::TimestepNotPositive:
      return "timestep not positive";
      break;
    case ErrorCodes::CostFunNotSet:
      return "cost function not set";
      break;
    case ErrorCodes::DynamicsFunNotSet:
      return "dynamics function not set";
      break;
    case ErrorCodes::InvalidOptAtTerminalKnotPoint:
      return "invalid operation at terminal knot point index";
      break;
    case ErrorCodes::MaxConstraintsExceeded:
      return "max number of constraints at a knot point exceeded.";
      break;
    case ErrorCodes::InvalidConstraintDim:
      return "invalid constraint dimension";
      break;
    case ErrorCodes::CholeskyFailed:
      return "Cholesky factorization failed";
      break;
    case ErrorCodes::FileError:
      return "file error";
      break;
    case ErrorCodes::OpOnlyValidAtTerminalKnotPoint:
      return "operation only valid at terminal knot point";
      break;
    case ErrorCodes::InvalidPointer:
      return "Invalid pointer";
      break;
    case ErrorCodes::BackwardPassFailed:
      return "Backward pass failed. Trying increasing regularization";
      break;
    case ErrorCodes::LineSearchFailed:
      return "Line search failed to find a point satisfying the Strong Wolfe Conditions";
      break;
    case ErrorCodes::MeritFunctionGradientTooSmall:
      return "Merit function gradient under `opts.tol_meritfun_gradient`. Aborting line search";
      break;
    case ErrorCodes::InvalidBoundConstraint:
      return "Invalid bound constraint. Make sure all upper bounds are greater than or equal to the lower bounds";
      break;
    case ErrorCodes::NonPositivePenalty:
      return "Penalty must be strictly positive";
      break;
    case ErrorCodes::CostNotQuadratic:
      return "Invalid operation. Cost function not quadratic";
      break;
  }
  return nullptr;
}

void PrintErrorCode(ErrorCodes err) {
  fmt::print("Got error code {}: {}\n", static_cast<int>(err), ErrorCodeToString(err));
}

}  // namespace altro