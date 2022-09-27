//
// Created by brian on 9/22/22.
//

#include "altro/solver/solver.hpp"
#include "gtest/gtest.h"

namespace altro {

TEST(SolverImpl, Constructor) {
  int N = 10;
  SolverImpl solver(N);
  EXPECT_FALSE(solver.data_.begin()->IsTerminalKnotPoint());
  EXPECT_TRUE((solver.data_.end() - 1)->IsTerminalKnotPoint());
}


}  // namespace altro