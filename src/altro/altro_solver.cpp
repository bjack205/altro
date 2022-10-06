//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altro_solver.hpp"

#include "Eigen/Dense"
#include "altrocpp_interface/altrocpp_interface.hpp"
#include "solver/solver.hpp"
#include "utils/formatting.hpp"
#include "altro/solver/internal_types.hpp"

namespace altro {

altro::ALTROSolver::ALTROSolver(int horizon_length)
    : solver_(std::make_unique<SolverImpl>(horizon_length)) {}

// ALTROSolver::ALTROSolver(const ALTROSolver& other)
//     : solver_(std::make_unique<SolverImpl>(*other.solver_)) {}
ALTROSolver::ALTROSolver(ALTROSolver &&other) = default;
// ALTROSolver& ALTROSolver::operator=(const ALTROSolver& other) {
//   *this->solver_ = *other.solver_;
//   return *this;
// }
ALTROSolver &ALTROSolver::operator=(ALTROSolver &&other) = default;
altro::ALTROSolver::~ALTROSolver() = default;

/////////////////////////////////////////////
// Setters
/////////////////////////////////////////////
ErrorCodes ALTROSolver::SetDimension(int num_states, int num_inputs, int k_start, int k_stop) {
  if (IsInitialized()) {
    ALTRO_THROW(
        AltroErrorException("Cannot change the dimension once the solver has been initialized.",
                            ErrorCodes::SolverAlreadyInitialized));
  }
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  if (err != ErrorCodes::NoError) return err;
  if (num_states <= 0) return ErrorCodes::StateDimUnknown;
  for (int k = k_start; k < k_stop; ++k) {
    solver_->nx_[k] = num_states;
    solver_->nu_[k] = num_inputs;
    err = solver_->data_[k].SetDimension(num_states, num_inputs);
    if (err != ErrorCodes::NoError) return err;  // TODO: reset to previous state on error?

    // Set prev next state dimension
    if (k > 0) {
      err = solver_->data_[k - 1].SetNextStateDimension(num_states);
    }
    if (err != ErrorCodes::NoError) return err;  // TODO: reset to previous state on error?
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetTimeStep(float h, int k_start, int k_stop) {
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Exclusive);
  if (err != ErrorCodes::NoError) return err;
  if (h <= 0.0f) {
    return ErrorCodes::TimestepNotPositive;
  }
  for (int k = k_start; k < k_stop; ++k) {
    err = solver_->data_[k].SetTimestep(h);
    if (err != ErrorCodes::NoError) return err;  // TODO: reset to previous state on error?

    // AltroCpp Interface
    solver_->h_[k] = h;
  }
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Set Dynamics
/////////////////////////////////////////////
ErrorCodes ALTROSolver::SetExplicitDynamics(ExplicitDynamicsFunction dynamics_function,
                                            ExplicitDynamicsJacobian dynamics_jacobian, int k_start,
                                            int k_stop) {
  ErrorCodes err = AssertDimensionsAreSet(k_stop, k_stop, "Cannot set the dynamics");
  err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Exclusive);
  if (err != ErrorCodes::NoError) return err;

  for (int k = k_start; k < k_stop; ++k) {
    int n = this->GetStateDim(k);
    int m = this->GetInputDim(k);
    err = solver_->data_[k].SetDynamics(dynamics_function, dynamics_jacobian);
    if (err != ErrorCodes::NoError) return err;

    // AltroCpp Interface
    cpp_interface::GeneralDiscreteDynamics dynamics(n, m, dynamics_function, dynamics_jacobian);
    solver_->problem_.SetDynamics(
        std::make_shared<cpp_interface::GeneralDiscreteDynamics>(dynamics), k);
  }
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Set Cost Function
/////////////////////////////////////////////
ErrorCodes ALTROSolver::SetCostFunction(CostFunction cost_function, CostGradient cost_gradient,
                                        CostHessian cost_hessian, int k_start, int k_stop) {
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  if (err != ErrorCodes::NoError) return err;

  for (int k = k_stop; k < k_stop; ++k) {
    solver_->data_[k].SetCostFunction(cost_function, cost_gradient, cost_hessian);
  }

  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetDiagonalCost(int num_states, int num_inputs, const a_float *Q_diag,
                                        const a_float *R_diag, const a_float *q, const a_float *r,
                                        a_float c, int k_start, int k_stop) {
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  err = AssertDimensionsAreSet(k_start, k_stop, "Cannot set the cost function");
  if (err != ErrorCodes::NoError) return err;

  for (int k = k_stop; k < k_stop; ++k) {
    int n = this->GetStateDim(k);
    int m = this->GetInputDim(k);
    if (n != num_states) return ErrorCodes::DimensionMismatch;
    if (k != GetHorizonLength() && m != num_inputs) return ErrorCodes::DimensionMismatch;
    solver_->data_[k].SetDiagonalCost(n, m, Q_diag, R_diag, q, r, c);

    // New interface
    solver_->data_[k].SetDiagonalCost(n, m, Q_diag, R_diag, q, r, c);
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetQuadraticCost(int num_states, int num_inputs, const a_float *Q,
                                         const a_float *R, const a_float *H, const a_float *q,
                                         const a_float *r, a_float c, int k_start, int k_stop) {
  using MatrixXd = Eigen::Matrix<a_float, Eigen::Dynamic, Eigen::Dynamic>;
  using VectorXd = Eigen::Vector<a_float, Eigen::Dynamic>;
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  err = AssertDimensionsAreSet(k_start, k_stop, "Cannot set the cost function");
  if (err != ErrorCodes::NoError) return err;

  for (int k = k_start; k < k_stop; ++k) {
    int n = this->GetStateDim(k);
    int m = this->GetInputDim(k);
    if (n != num_states) return ErrorCodes::DimensionMismatch;
    if (k != GetHorizonLength() && m != num_inputs) return ErrorCodes::DimensionMismatch;
    solver_->data_[k].SetQuadraticCost(n, m, Q, R, H, q, r, c);

    // AltroCpp interface
    MatrixXd Qmat = Eigen::Map<const MatrixXd>(Q, n, n);
    MatrixXd Rmat = Eigen::Map<const MatrixXd>(R, m, m);
    MatrixXd Hmat = Eigen::Map<const MatrixXd>(H, m, n);
    VectorXd qvec = Eigen::Map<const VectorXd>(q, n);
    VectorXd rvec = Eigen::Map<const VectorXd>(r, m);
    bool is_terminal = k == this->GetHorizonLength();
    cpp_interface::QuadraticCost cost(Qmat, Rmat, Hmat, qvec, rvec, c, is_terminal);
    solver_->problem_.SetCostFunction(std::make_shared<cpp_interface::QuadraticCost>(cost), k);

    // New interface
    solver_->data_[k].SetQuadraticCost(n, m, Q, R, H, q, r, c);
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetLQRCost(int num_states, int num_inputs, const a_float *Q_diag,
                                   const a_float *R_diag, const a_float *x_ref,
                                   const a_float *u_ref, int k_start, int k_stop) {
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  err = AssertDimensionsAreSet(k_start, k_stop, "Cannot set the cost function");
  if (err != ErrorCodes::NoError) return err;

  for (int k = k_start; k < k_stop; ++k) {
    int n = this->GetStateDim(k);
    int m = this->GetInputDim(k);
    if (n != num_states) return ErrorCodes::DimensionMismatch;
    if (k != GetHorizonLength() && m != num_inputs) return ErrorCodes::DimensionMismatch;
    Eigen::Map<const Vector> Qd(Q_diag, n);
    Eigen::Map<const Vector> Rd(R_diag, m);
    Eigen::Map<const Vector> xref(x_ref, n);
    Eigen::Map<const Vector> uref(u_ref, m);
    Vector q = -(Qd.asDiagonal() * xref);
    Vector r = -(Rd.asDiagonal() * uref);
    a_float c = 0.5 * xref.transpose() * Qd.asDiagonal() * xref;
    c += 0.5 * uref.transpose() * Rd.asDiagonal() * uref;
    solver_->data_[k].SetDiagonalCost(n, m, Q_diag, R_diag, q.data(), r.data(), c);

    // AltroCpp interface
    Matrix Qmat = Eigen::Map<const Vector>(Q_diag, n).asDiagonal();
    Matrix Rmat = Eigen::Map<const Vector>(R_diag, m).asDiagonal();
    bool is_terminal = k == this->GetHorizonLength();
    cpp_interface::QuadraticCost cost =
        cpp_interface::QuadraticCost::LQRCost(Qmat, Rmat, xref.eval(), uref.eval(), is_terminal);
    solver_->problem_.SetCostFunction(std::make_shared<cpp_interface::QuadraticCost>(cost), k);

  }
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Other Setters
/////////////////////////////////////////////
ErrorCodes ALTROSolver::SetInitialState(const double *x0, int n) {
  int n0 = GetStateDim(0);
  bool first_state_dim_is_set = n0 > 0;
  if (!first_state_dim_is_set) {
    solver_->nx_[0] = n;
  } else if (n != n0) {
    ALTRO_THROW(
        AltroErrorException(fmt::format("Dimension mismatch: The provided state dimension was "
                                        "{}, but was previously set to {}.",
                                        n, n0),
                            ErrorCodes::DimensionMismatch));
  }
  solver_->initial_state_ = Eigen::Map<const VectorXd>(x0, n);

  // AltroCpp Interface
  VectorXd x0_vec = Eigen::Map<const VectorXd>(x0, n);
  solver_->problem_.SetInitialState(x0_vec);
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetConstraint(ConstraintFunction constraint_function,
                                      ConstraintJacobian constraint_jacobian, int dim,
                                      ConstraintType constraint_type, std::string label,
                                      int k_start, int k_stop,
                                      std::vector<ConstraintIndex> *con_inds) {
  ErrorCodes err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  err = AssertDimensionsAreSet(k_start, k_stop, "Cannot set constraint");
  if (err != ErrorCodes::NoError) return err;
  int num_indices = k_stop - k_start;
  if (con_inds) con_inds->reserve(num_indices);

  for (int k = k_start; k < k_stop; ++k) {
    int n = GetStateDim(k);
    int m = GetInputDim(k);
    std::string label_k = label;
    if (num_indices != 1) {
      label_k += "_" + std::to_string(k);
    }

    // Add constraint to problem
    int ncon = -1;
    if (constraint_type == ConstraintType::EQUALITY) {
      cpp_interface::EqualityConstraint eq(n, m, dim, constraint_function, constraint_jacobian,
                                           label_k);
      solver_->problem_.SetConstraint(
          std::make_shared<cpp_interface::EqualityConstraint>(std::move(eq)), k);
      ncon = solver_->problem_.GetNumEqualityConstraints(k);
    } else if (constraint_type == ConstraintType::INEQUALITY) {
      cpp_interface::InequalityConstraint ineq(n, m, dim, constraint_function, constraint_jacobian,
                                               label_k);
      solver_->problem_.SetConstraint(
          std::make_shared<cpp_interface::InequalityConstraint>(std::move(ineq)), k);
      ncon = solver_->problem_.GetNumInequalityConstraints(k);
    }

    // Set index
    ConstraintIndex idx(k, ncon);
    if (con_inds) con_inds->emplace_back(idx);
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::Initialize() {
  AssertDimensionsAreSet(0, GetHorizonLength(), "Cannot initialize solver");
  AssertTimestepsArePositive("Cannot initialize solver");
  return solver_->Initialize();
}

ErrorCodes ALTROSolver::SetState(const a_float *x, int n, int k_start, int k_stop) {
  ErrorCodes err = AssertInitialized();
  err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Inclusive);
  if (err != ErrorCodes::NoError) return err;
  for (int k = k_start; k < k_stop; ++k) {
    AssertStateDim(k, n);
    solver_->data_[k].x_ = Eigen::Map<const Vector>(x, n);

    // AltroCpp Interface
    solver_->trajectory_->State(k) = Eigen::Map<const Vector>(x, n);
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::SetInput(const a_float *u, int m, int k_start, int k_stop) {
  ErrorCodes err = AssertInitialized();
  err = CheckKnotPointIndices(k_start, k_stop, LastIndexMode::Exclusive);
  if (err != ErrorCodes::NoError) return err;
  for (int k = k_start; k < k_stop; ++k) {
    AssertInputDim(k, m);
    solver_->data_[k].u_ = Eigen::Map<const Vector>(u, m);

    // AltroCpp Interface
    solver_->trajectory_->Control(k) = Eigen::Map<const Vector>(u, m);
  }
  return ErrorCodes::NoError;
}

void ALTROSolver::SetOptions(const AltroOptions &opts) { solver_->opts = opts; }

SolveStatus ALTROSolver::Solve() {
  solver_->Solve();
  return solver_->stats.status;
}

/***************************************************************************************************
 * Getters
 ***************************************************************************************************/

int ALTROSolver::GetHorizonLength() const { return solver_->horizon_length_; }

int ALTROSolver::GetStateDim(int k) const { return solver_->data_[k].GetStateDim(); }

int ALTROSolver::GetInputDim(int k) const { return solver_->data_[k].GetInputDim(); }

float ALTROSolver::GetTimeStep(int k) const { return solver_->h_[k]; }

bool ALTROSolver::IsInitialized() const { return solver_->IsInitialized(); }

AltroOptions &ALTROSolver::GetOptions() { return solver_->opts; }

const AltroOptions &ALTROSolver::GetOptions() const { return solver_->opts; }

a_float ALTROSolver::CalcCost() { return solver_->CalcCost(); }

int ALTROSolver::GetIterations() const { return solver_->stats.iterations; }

a_float ALTROSolver::GetSolveTimeMs() const { return solver_->stats.solve_time.count(); }

a_float ALTROSolver::GetPrimalFeasibility() const { return solver_->stats.primal_feasibility; };

a_float ALTROSolver::GetFinalObjective() const { return solver_->stats.objective_value; }

ErrorCodes ALTROSolver::GetState(a_float *x, int k) const {
  int k_stop = k + 1;
  ErrorCodes err = CheckKnotPointIndices(k, k_stop, LastIndexMode::Exclusive);
  err = AssertDimensionsAreSet(k, k_stop);
  if (err != ErrorCodes::NoError) return err;

  int n = GetStateDim(k);
  Eigen::Map<Eigen::VectorXd>(x, n) = solver_->data_[k].x;
//  Eigen::Map<Eigen::VectorXd>(x, n) = solver_->trajectory_->State(k);
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::GetInput(a_float *u, int k) const {
  int k_stop = k + 1;
  ErrorCodes err = CheckKnotPointIndices(k, k_stop, LastIndexMode::Exclusive);
  err = AssertDimensionsAreSet(k, k_stop);
  if (err != ErrorCodes::NoError) return err;

  int m = GetInputDim(k);
  Eigen::Map<Eigen::VectorXd>(u, m) = solver_->data_[k].u;
//  Eigen::Map<Eigen::VectorXd>(u, m) = solver_->trajectory_->Control(k);
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::GetDualDynamics(a_float *y, int k) const {
  int k_stop = k + 1;
  ErrorCodes err = CheckKnotPointIndices(k, k_stop, LastIndexMode::Exclusive);
  err = AssertDimensionsAreSet(k, k_stop);
  if (err != ErrorCodes::NoError) return err;

  int n = GetStateDim(k);
  Eigen::Map<Eigen::VectorXd>(y, n) = solver_->data_[k].y;
  return ErrorCodes::NoError;
}

/***************************************************************************************************
 * Private methods
 ***************************************************************************************************/
ErrorCodes ALTROSolver::AssertInitialized() const {
  if (!this->IsInitialized()) {
    ALTRO_THROW(
        AltroErrorException("Solver must be initialized.", ErrorCodes::SolverNotInitialized));
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::AssertDimensionsAreSet(int k_start, int k_stop, std::string msg) const {
  int N = GetHorizonLength();
  for (int k = k_start; k < k_stop; ++k) {
    int n = GetStateDim(k);
    int m = GetInputDim(k);
    if (n <= 0 || (k < N && m <= 0)) {
      ALTRO_THROW(AltroErrorException(
          fmt::format("{}. Dimensions at knotpoint {} haven't been set.", msg, k),
          ErrorCodes::DimensionUnknown));
    }
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::CheckKnotPointIndices(int k_start, int &k_stop,
                                              LastIndexMode last_index) const {
  // Get upper index range
  int terminal_index = 0;
  switch (last_index) {
    case LastIndexMode::Inclusive:
      terminal_index = GetHorizonLength();
      break;
    case LastIndexMode::Exclusive:
      terminal_index = GetHorizonLength() - 1;
      break;
  }

  // Automatic full range selection
  if (k_start == 0 && k_stop == LastIndex) {
    k_start = 0;
    k_stop = terminal_index + 1;
  }

  // Default to index if stopping index is unset
  if (k_stop <= 0) k_stop = k_start + 1;

  // Check indices are valid
  int N = GetHorizonLength();
  if (k_start < 0 || k_start > terminal_index) {
    // TODO: use custom exception type
    ALTRO_THROW(AltroErrorException(
        fmt::format("Knot point index out of range. Should be in range {} - {}, got {}.", 0, N,
                    k_start),
        ErrorCodes::BadIndex));
  }
  if (k_stop < 0 || k_start > terminal_index + 1) {
    ALTRO_THROW(AltroErrorException(
        fmt::format("Terminal knot point index out of range. Should be in range {} - {}, got {}.",
                    0, N + 1, k_stop),
        ErrorCodes::BadIndex))
  }
  if (k_stop > 0 && k_stop <= k_start) {
    // clang-format off
    fmt::print("WARNING [ALTRO]: Stopping index {} not greater than starting index {}. Index range is empty.\n", k_stop, k_stop);
    fmt::print("                 To set a single index, set k_stop = k_start + 1.");
    // clang-format on
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::AssertStateDim(int k, int n) const {
  int state_dim = GetStateDim(k);
  if (state_dim != n) {
    ALTRO_THROW(AltroErrorException(
        fmt::format("State dimension mismatch. Got {}, expected {}.", n, state_dim),
        ErrorCodes::DimensionMismatch));
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::AssertInputDim(int k, int m) const {
  int input_dim = GetInputDim(k);
  if (input_dim != m) {
    ALTRO_THROW(AltroErrorException(
        fmt::format("Input dimension mismatch. Got {}, expected {}.", m, input_dim),
        ErrorCodes::DimensionMismatch));
  }
  return ErrorCodes::NoError;
}

ErrorCodes ALTROSolver::AssertTimestepsArePositive(std::string msg) const {
  for (int k = 0; k < GetHorizonLength(); ++k) {
    float h = GetTimeStep(k);
    if (h <= 0.0) {
      ALTRO_THROW(
          AltroErrorException(fmt::format("{}. Timestep is nonpositive at timestep {}.\n", msg, k),
                              ErrorCodes::NonPositive));
    }
  }
  return ErrorCodes::NoError;
}

void ALTROSolver::PrintStateTrajectory() const {
  fmt::print("STATE TRAJECTORY:\n");
  for (int k = 0; k <= GetHorizonLength(); ++k) {
    fmt::print(" x[{:03d}]: [{}]\n", k, solver_->data_[k].x_.transpose().eval());
  }
}

void ALTROSolver::PrintInputTrajectory() const {
  fmt::print("INPUT TRAJECTORY:\n");
  for (int k = 0; k < GetHorizonLength(); ++k) {
    fmt::print(" u[{:03d}]: [{}]\n", k, solver_->data_[k].u_.transpose().eval());
  }
}

}  // namespace altro
