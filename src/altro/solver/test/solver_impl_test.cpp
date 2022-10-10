//
// Created by brian on 9/22/22.
//

#include "altro/altro_solver.hpp"
#include "altro/solver/solver.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

namespace altro {

class SolverImplTest : public ::testing::Test {
 protected:
  SolverImplTest() : solver(10) {}

  void SetUp() override {
  }

  void InitializeDoubleIntegratorSolver() {
    const int dim = 2;
    n = 2 * dim;
    m = dim;
    N = 10;
    h = 0.01;

    // Equilibrium
    Vector xeq(n);
    Vector ueq(m);
    xeq << 1, 2, 0, 0;
    ueq << 0,0;

    // Initial state
    x0 = Vector::Zero(n);
    x0 << 10.5, -20.5, -4, 5;

    // Calculate A,B,f
    Matrix jac(n, n + m);
    Vector f = Vector::Zero(n);

    discrete_double_integrator_dynamics(f.data(), xeq.data(), ueq.data(), h, dim);
    discrete_double_integrator_jacobian(jac.data(), xeq.data(), ueq.data(), h, dim);
    Matrix A = jac.leftCols(n);
    Matrix B = jac.rightCols(m);

    // Cost
    Vector Qd = Vector::Constant(n, 1.1);
    Vector Rd = Vector::Constant(m, 0.1);
    Vector Qfd = Qd * 100;
    Matrix Q = Qd.asDiagonal();
    Matrix R = Rd.asDiagonal();
    Matrix Qf = Qfd.asDiagonal();
    Vector q = Vector::Constant(n, 0.01);
    Vector r = Vector::Constant(m, 0.001);
    float c = 0.0;

    // Initialize solver
    ErrorCodes err;
    solver.initial_state_ = x0;
    for (int k = 0; k < N; ++k) {
      err = solver.data_[k].SetDimension(n, m);
      EXPECT_EQ(err, ErrorCodes::NoError);

      err = solver.data_[k].SetNextStateDimension(n);
      EXPECT_EQ(err, ErrorCodes::NoError);

      err = solver.data_[k].SetTimestep(h);
      EXPECT_EQ(err, ErrorCodes::NoError);

      err = solver.data_[k].SetLinearDynamics(n, n, m, A.data(), B.data(), f.data());
      EXPECT_EQ(err, ErrorCodes::NoError);

      err = solver.data_[k].SetDiagonalCost(n, m, Qd.data(), Rd.data(), q.data(), r.data(), c);
      EXPECT_EQ(err, ErrorCodes::NoError);

      err = solver.data_[k].Initialize();
      EXPECT_EQ(err, ErrorCodes::NoError);
      EXPECT_TRUE(solver.data_[k].IsInitialized());
    }
    err = solver.data_[N].SetDimension(n, 0);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[N].SetDiagonalCost(n, 0, Qfd.data(), nullptr, q.data(), nullptr, 0.0);
    EXPECT_EQ(err, ErrorCodes::NoError);

    err = solver.data_[N].Initialize();
    EXPECT_EQ(err, ErrorCodes::NoError);

    EXPECT_TRUE(solver.data_[N].CostFunctionIsQuadratic());
    EXPECT_TRUE(solver.data_[N].IsInitialized());
  }



  SolverImpl solver;
  int n;
  int m;
  int N;
  float h;
  Vector x0;

};

TEST(SolverImpl, Constructor) {
  int N = 10;
  SolverImpl solver(N);
  EXPECT_FALSE(solver.data_.begin()->IsTerminalKnotPoint());
  EXPECT_TRUE((solver.data_.end() - 1)->IsTerminalKnotPoint());
}

TEST_F(SolverImplTest, TVLQR_Test) {
  fmt::print("\n#############################################\n");
  fmt::print("                TVLQR Test\n");
  fmt::print("#############################################\n");
  InitializeDoubleIntegratorSolver();
  solver.Initialize();

  // Calculate backward pass
  solver.BackwardPass();
  solver.BackwardPass();
  Matrix K0_expected(m,n);
  Vector d0_expected(m);
  // clang-format off
  K0_expected << 0.7753129718046554, 0.0, 5.840445640045901,
                 0.0, 0.0, 0.7753129718046554, 0.0, 5.840445640045901;
  d0_expected << -7.634078625343007, -15.256221385516275;
  // clang-format on

  // not sure why they're not matching to machine precision
  double K_err = (solver.data_[0].K_ - K0_expected).lpNorm<Eigen::Infinity>();
  double d_err = (solver.data_[0].d_ - d0_expected).lpNorm<Eigen::Infinity>();
  fmt::print("K error {}\n", K_err);
  fmt::print("d error {}\n", d_err);

  EXPECT_LT((solver.data_[0].K_ - K0_expected).norm(), 1e-6);
  EXPECT_LT((solver.data_[0].d_ - d0_expected).norm(), 1e-6);

  // Linear rollout
  solver.LinearRollout();
  Vector xN(n);
  Vector yN(n);
  xN << 20.165445369740308, -0.13732391651279308, -2.3724421496097037, 2.3113121303468707;
  yN << 2218.2089906714345, -15.09563081640724, -260.9586364570674, 254.2543343381558;
  double x_err = (xN - solver.data_[N].x_).lpNorm<Eigen::Infinity>();
  double y_err = (yN - solver.data_[N].y_).lpNorm<Eigen::Infinity>();
  fmt::print("x error {}\n", x_err);
  fmt::print("y error {}\n", y_err);
  EXPECT_LT(x_err, 1e-6);
  EXPECT_LT(y_err, 1e-5);

  // Check stationarity
  solver.CalcCostGradient();
  a_float stat = solver.Stationarity();
  fmt::print("Stationarity: {}\n", stat);
  EXPECT_LT(stat, 1e-10);
}

TEST_F(SolverImplTest, CopyTrajectory) {
  InitializeDoubleIntegratorSolver();
  solver.Initialize();

  // Set the initial trajectory
  std::vector<Vector> xref(N+1);
  std::vector<Vector> uref(N);
  Vector xf = Vector::Constant(n, 10.0);
  for (int k = 0; k < N; ++k) {
    double theta = k / static_cast<double>(N);
    xref[k] = x0 + (xf - x0) * theta;
    uref[k] = Vector::Constant(m, theta);

    solver.data_[k].x_ = xref[k];
    solver.data_[k].u_ = uref[k];
  }
  xref[N] = xf;
  solver.data_[N].x_ = xref[N];

  // Copy trajectory to reference
  EXPECT_GT((solver.data_[N-1].x - solver.data_[N-1].x_).norm(), 0.1);
  EXPECT_GT((solver.data_[N-1].u - solver.data_[N-1].u_).norm(), 0.1);
  EXPECT_GT((solver.data_[N].x - solver.data_[N].x_).norm(), 0.1);
  solver.CopyTrajectory();
  EXPECT_LT((solver.data_[N-1].x - solver.data_[N-1].x_).norm(), 1e-10);
  EXPECT_LT((solver.data_[N-1].u - solver.data_[N-1].u_).norm(), 1e-10);
  EXPECT_LT((solver.data_[N].x - solver.data_[N].x_).norm(), 1e-10);
}

TEST_F(SolverImplTest, MeritFunTest) {
  fmt::print("\n#############################################\n");
  fmt::print("                Merit Fun Test\n");
  fmt::print("#############################################\n");
  InitializeDoubleIntegratorSolver();
  solver.Initialize();

  // Set the initial trajectory
  std::vector<Vector> xref(N+1);
  std::vector<Vector> uref(N);
  Vector xf = Vector::Zero(n);
  xf << -1,2,0,0;
  for (int k = 0; k < N; ++k) {
    double theta = k / static_cast<double>(N);
    xref[k] = x0 + (xf - x0) * theta;
    uref[k] = Vector::Constant(m, theta);

    solver.data_[k].x_ = xref[k];
    solver.data_[k].u_ = uref[k];
  }
  xref[N] = xf;
  solver.data_[N].x_ = xref[N];

  // Copy trajectory to reference
  solver.CopyTrajectory();
  EXPECT_LT((solver.data_[0].x - x0).norm(), 1e-10);
  EXPECT_LT((solver.data_[N].x - xf).norm(), 1e-10);

  // Compute gains
  for (int k = 0; k <= N; ++k) {
    solver.data_[k].CalcCostGradient();
    solver.data_[k].CalcConstraints();
    solver.data_[k].CalcConstraintJacobians();
    solver.data_[k].CalcProjectedDuals();
    solver.data_[k].CalcConicJacobians();
    solver.data_[k].CalcDynamicsExpansion();
  }
  solver.CalcExpansions();
  solver.BackwardPass();

  // Calculate the merit function
  ErrorCodes err;
  a_float alpha = 1.0;
  a_float phi = 0.0;
  a_float dphi = 0.0;
  err = solver.MeritFunction(alpha, &phi, &dphi);
  EXPECT_EQ(err, ErrorCodes::NoError);
  a_float phi0 = phi;
  a_float dphi0 = dphi;

  const double phi_expected = 25992.822836536347;
  const double dphi_expected = -43.52330058003784;
  a_float phi_err = std::abs(phi - phi_expected) / std::abs(phi_expected);
  a_float dphi_err = std::abs(dphi - dphi_expected) / std::abs(dphi_expected);
  EXPECT_LT(phi_err, 1e-6);
  EXPECT_LT(dphi_err, 1e-6);

  fmt::print("alpha = 1\n");
  fmt::print("  phi err  = {}\n", phi_err);
  fmt::print("  dphi err = {}\n", dphi_err);

  // Finite Diff check
  double eps = 1e-6;
  a_float phi1;
  alpha = 1.0;
  solver.MeritFunction(alpha + eps, &phi1, nullptr);
  double dphi_finite_diff = (phi1 - phi0) / eps;
  double dphi_err_finite_diff = std::abs(dphi0 - dphi_finite_diff) / std::abs(dphi0);
  fmt::print("  dphi err (finite diff) = {}\n", dphi_err_finite_diff);
  EXPECT_LT(dphi_err_finite_diff, 1e-6);

  // Try at alpha = 0
  alpha = 0;
  err = solver.MeritFunction(alpha, &phi, &dphi);
  EXPECT_EQ(err, ErrorCodes::NoError);
  const double phi0_expected = 26039.092492842017;
  const double dphi0_expected = -49.01601203132092;
  phi_err = std::abs(phi - phi0_expected) / std::abs(phi0_expected);
  dphi_err = std::abs(dphi - dphi0_expected) / std::abs(dphi0_expected);
  EXPECT_LT(phi_err, 1e-6);
  EXPECT_LT(dphi_err, 1e-6);

  fmt::print("alpha = 0\n");
  fmt::print("  phi err  = {}\n", phi_err);
  fmt::print("  dphi err = {}\n", dphi_err);
}

TEST_F(SolverImplTest, ForwardPassTest) {
  fmt::print("\n#############################################\n");
  fmt::print("                Forward Pass\n");
  fmt::print("#############################################\n");
  InitializeDoubleIntegratorSolver();
  solver.Initialize();

  // Set the initial trajectory
  std::vector<Vector> uref(N);
  for (int k = 0; k < N; ++k) {
    double theta = k / static_cast<double>(N);
    uref[k] = Vector::Constant(m, theta);

    solver.data_[k].u_ = uref[k];
  }
  solver.OpenLoopRollout();

  // Copy trajectory to reference
  solver.CopyTrajectory();
  EXPECT_LT((solver.data_[0].x - x0).norm(), 1e-10);
//  EXPECT_LT((solver.data_[N].x - xf).norm(), 1e-10);

  // Compute gains
  for (int k = 0; k <= N; ++k) {
    solver.data_[k].CalcCostGradient();
    solver.data_[k].CalcConstraints();
    solver.data_[k].CalcConstraintJacobians();
    solver.data_[k].CalcProjectedDuals();
    solver.data_[k].CalcConicJacobians();
    solver.data_[k].CalcDynamicsExpansion();
  }
  solver.CalcExpansions();
  solver.BackwardPass();

  // Check that the merit function has zero slow at a step of 1.0
  double phi, dphi;
  solver.MeritFunction(1.0, &phi, &dphi);
  EXPECT_LT(std::abs(dphi), 1e-8);

  // Run ForwardPass
  double alpha;
  solver.ForwardPass(&alpha);
  EXPECT_DOUBLE_EQ(alpha, 1.0);
}


}  // namespace altro