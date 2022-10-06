//
// Created by Brian Jackson on 9/7/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "Eigen/Dense"
#include "fmt/core.h"
#include "fmt/ostream.h"

template <class T, int N1, int N2>
struct fmt::formatter<Eigen::Matrix<T, N1, N2>> : public fmt::ostream_formatter {};

template <class T, int N1, int N2>
struct fmt::formatter<Eigen::WithFormat<Eigen::Matrix<T, N1, N2>>> : public fmt::ostream_formatter {
};

namespace altro {
template <class T>
void PrintVectorRow(std::string label, T x) {
  fmt::print("{}[{}]\n", label, x.transpose().eval());
}
}  // namespace altro

