//
// Created by Brian Jackson on 9/25/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <stdexcept>

#undef ALTRO_THROW
#ifdef ALTRO_ENABLE_RUNTIME_EXCEPTIONS
#define ALTRO_THROW(x) throw(x)
#else
#define ALTRO_THROW(x) return ((x).Errno());
#endif

enum class ErrorCodes {
  NoError,
  DimensionUnknown,
  BadIndex,
  DimensionMismatch,
  SolverNotInitialized,
  SolverAlreadyInitialized,
  NonPositive,
  FileError,
};

class AltroErrorException : public std::runtime_error {
 public:
  AltroErrorException(std::string msg, ErrorCodes code)
      : std::runtime_error(msg.c_str()), code_(code) {}

  virtual ErrorCodes Errno() { return code_; }

  virtual ~AltroErrorException(){};

 private:
  ErrorCodes code_;
};
