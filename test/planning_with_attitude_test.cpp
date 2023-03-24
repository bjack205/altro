//
// Created by Zixin Zhang on 3/24/23
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include <chrono>
#include <iostream>
#include <filesystem>

#include "Eigen/Dense"
#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "altro/utils/formatting.hpp"
#include "fmt/core.h"
#include "fmt/chrono.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "nlohmann/json.hpp"

using Eigen::MatrixXd;

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace altro;

TEST(PlanningWithAttitudeTest, Dynamics) {

}