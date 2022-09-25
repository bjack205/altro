#include "altro/altro.hpp"
#include "fmt/core.h"
#include "Eigen/Dense"

int main() {
  altro::ALTROSolver solver(10);
  solver.SetDimension(4, 2, 0, altro::LastIndex);
  fmt::print("Solver Initialized!\n");
  return 0;
}