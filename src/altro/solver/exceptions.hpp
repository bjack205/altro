//
// Created by Brian Jackson on 9/25/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <stdexcept>

#include "fmt/core.h"
#include "fmt/color.h"

#undef ALTRO_THROW
#ifdef ALTRO_ENABLE_RUNTIME_EXCEPTIONS
#define ALTRO_THROW(msg, code) \
  (throw(AltroErrorException((msg), code)), code)
#else
#define ALTRO_THROW(msg, code) \
  (fmt::print(fmt::fg(fmt::color::red), "ALTRO ERROR Code {}: {} {}:{}\n  Message: {}\n", (static_cast<int>(code)), ErrorCodeToString(code), __FILE__, __LINE__, (msg)), code)
#endif

namespace altro {

enum class ErrorCodes {
  NoError,
  StateDimUnknown,
  InputDimUnknown,
  NextStateDimUnknown,
  DimensionUnknown,
  BadIndex,
  DimensionMismatch,
  SolverNotInitialized,
  SolverAlreadyInitialized,
  NonPositive,
  TimestepNotPositive,
  CostFunNotSet,
  DynamicsFunNotSet,
  InvalidOpAtTeriminalKnotPoint,
  MaxConstraintsExceeded,
  InvalidConstraintDim,
  CholeskyFailed,
  OpOnlyValidAtTerminalKnotPoint,
  InvalidPointer,
  BackwardPassFailed,
  LineSearchFailed,
  MeritFunctionGradientTooSmall,
  InvalidBoundConstraint,
  FileError,
};

void PrintErrorCode(ErrorCodes err);

const char *ErrorCodeToString(ErrorCodes err);

class AltroErrorException : public std::runtime_error {
 public:
  AltroErrorException(std::string msg, ErrorCodes code)
      : std::runtime_error(msg.c_str()), code_(code) {}

  virtual ErrorCodes Errno() { return code_; }

  virtual ~AltroErrorException(){};

 private:
  ErrorCodes code_;
};

}  // namespace altro