//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "knotpoint_data.hpp"

namespace altro {

/////////////////////////////////////////////
// Constructor
/////////////////////////////////////////////
KnotPointData::KnotPointData(bool is_terminal) : is_terminal_(is_terminal) {}


/////////////////////////////////////////////
// Setters
/////////////////////////////////////////////
ErrorCodes KnotPointData::SetDimension(int num_states, int num_inputs) {
  if (num_states <= 0) {
    return ErrorCodes::StateDimUnknown;
  }
  if (!IsTerminalKnotPoint() && num_inputs <= 0) {
    return ErrorCodes::InputDimUnknown;
  }
  num_states_ = num_states;
  if (!IsTerminalKnotPoint()) {
    num_inputs_ = num_inputs;
  }
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetNextStateDimension(int num_states_next) {
  if (IsTerminalKnotPoint()) {
    return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  }
  if (num_states_next <= 0) {
    return ErrorCodes::StateDimUnknown;
  }
  num_next_state_ = num_states_next;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetTimestep(float h) {
  if (IsTerminalKnotPoint()) {
    return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  }
  if (h <= 0.0f) {
    return ErrorCodes::TimestepNotPositive;
  }
  h_ = h;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetQuadraticCost(int n, int m, const a_float *Qmat, const a_float *Rmat,
                                           const a_float *Hmat, const a_float *q, const a_float *r,
                                           a_float c) {
  if (num_states_ > 0 && num_states_ != n) return ErrorCodes::DimensionMismatch;
  if (num_inputs_ > 0 && num_inputs_ != m) return ErrorCodes::DimensionMismatch;

  Q_ = Vector::Zero(n * n);
  R_ = Vector::Zero(m * m);
  H_ = Matrix ::Zero(m, n);
  q_ = Eigen::Map<const Vector>(q, n);
  r_ = Eigen::Map<const Vector>(r, m);
  c_ = c;

  // Copy to expansion
  Q_.reshaped(n, n) = Eigen::Map<const Matrix>(Qmat, n, n);
  R_.reshaped(m, m) = Eigen::Map<const Matrix>(Rmat, m, m);
  H_.reshaped(m, n) = Eigen::Map<const Matrix>(Hmat, m, n);

  cost_fun_is_set_ = true;
  cost_fun_type_ = CostFunType::Quadratic;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetDiagonalCost(int n, int m, const a_float *Qdiag, const a_float *Rdiag,
                                          const a_float *q, const a_float *r, a_float c) {
  if (num_states_ > 0 && num_states_ != n) return ErrorCodes::DimensionMismatch;
  if (num_inputs_ > 0 && num_inputs_ != m) return ErrorCodes::DimensionMismatch;

  // Note: Assigns the first elements of the matrix to be the diagonal (to keep the same data size
  //       with a quadratic cost
  Q_ = Vector::Zero(n * n);
  Q_.head(n) = Eigen::Map<const Vector>(Qdiag, n);

  H_ = Matrix ::Zero(m, n);
  q_ = Eigen::Map<const Vector>(q, n);
  c_ = c;


  if (!IsTerminalKnotPoint()) {
    R_ = Vector::Zero(m * m);
    R_.head(m) = Eigen::Map<const Vector>(Rdiag, m);
    r_ = Eigen::Map<const Vector>(r, m);
  }

  cost_fun_is_set_ = true;
  cost_fun_type_ = CostFunType::Diagonal;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetCostFunction(CostFunction cost_function, CostGradient cost_gradient,
                                          CostHessian cost_hessian) {
  cost_function_ = std::move(cost_function);
  cost_gradient = std::move(cost_gradient);
  cost_hessian = std::move(cost_hessian);

  cost_fun_is_set_ = true;
  cost_fun_type_ = CostFunType::Generic;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetLinearDynamics(int n2, int n, int m, const altro::a_float *A, const altro::a_float *B, const altro::a_float *f) {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  if (num_states_ > 0 && n != num_states_) return ErrorCodes::DimensionMismatch;
  if (num_inputs_ > 0 && m != num_inputs_) return ErrorCodes::DimensionMismatch;
  if (num_next_state_ > 0 && n2 != num_next_state_) return ErrorCodes::DimensionMismatch;

  A_ = Eigen::Map<const Matrix>(A, n2, n);
  B_ = Eigen::Map<const Matrix>(B, n2, m);
  f_ = Eigen::Map<const Vector>(f, n2);

  dynamics_is_set_ = true;
  dynamics_are_linear_ = true;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetDynamics(ExplicitDynamicsFunction dynamics_function,
                                      ExplicitDynamicsJacobian dynamics_jacobian) {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  dynamics_function_ = std::move(dynamics_function);
  dynamics_jacobian_ = std::move(dynamics_jacobian);

  dynamics_is_set_ = true;
  dynamics_are_linear_ = false;
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::SetConstraint(ConstraintFunction constraint_function,
                                        ConstraintJacobian constraint_jacobian, int dim,
                                        ConstraintType constraint_type, std::string label) {
  if (NumConstraints() == kMaxConstraints) {
    return ErrorCodes::MaxConstraintsExceeded;
  }

  // QUESTION: should we allow constraints with 0 length?
  if (dim <= 0) {
    return ErrorCodes::InvalidConstraintDim;
  }

  constraint_function_.emplace_back(std::move(constraint_function));
  constraint_jacobian_.emplace_back(std::move(constraint_jacobian));
  constraint_dims_.emplace_back(dim);
  constraint_type_.emplace_back(std::move(constraint_type));
  constraint_label_.emplace_back(std::move(label));

  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Initialization
/////////////////////////////////////////////
ErrorCodes KnotPointData::Initialize() {
  const int n2 = GetNextStateDim();
  const int n = GetStateDim();
  const int m = GetInputDim();
  const float h = GetTimeStep();
  bool is_terminal = IsTerminalKnotPoint();

  // Check if the knot point can be initialized
  bool next_state_dim_is_set = n2 > 0;
  bool statedim_is_set = n > 0;
  bool inputdim_is_set = m > 0;
  bool timestep_is_set = h > 0.0f;

  if (!statedim_is_set) return ErrorCodes::StateDimUnknown;
  if (!is_terminal) {
    if (!inputdim_is_set) return ErrorCodes::InputDimUnknown;
    if (!next_state_dim_is_set) return ErrorCodes::NextStateDimUnknown;
    if (!timestep_is_set) return ErrorCodes::TimestepNotPositive;
    if (!dynamics_is_set_) return ErrorCodes::DynamicsFunNotSet;
  }
  if (!cost_fun_is_set_) return ErrorCodes::CostFunNotSet;

  // General data
  x = Vector::Zero(n);
  u = Vector::Zero(m);

  x_ = Vector::Zero(n);
  u_ = Vector::Zero(m);

  dynamics_jac_ = Matrix::Zero(n2, n + m);
  dynamics_val_ = Vector::Zero(n2);
  dynamics_dual_ = Vector::Zero(n2);

  c_x_hi_ = Vector::Zero(n);
  c_x_lo_ = Vector::Zero(n);
  c_u_hi_ = Vector::Zero(m);
  c_u_lo_ = Vector::Zero(m);

  v_x_hi_ = Vector::Zero(n);
  v_x_lo_ = Vector::Zero(n);
  v_u_hi_ = Vector::Zero(m);
  v_u_lo_ = Vector::Zero(m);

  // Constraint data
  int num_constraints = NumConstraints();
  constraint_jac_.reserve(num_constraints);
  constraint_val_.reserve(num_constraints);
  constraint_dual_.reserve(num_constraints);
  for (int i = 0; i < num_constraints; ++i) {
    int p = constraint_dims_[i];
    constraint_jac_.emplace_back(Matrix::Zero(p, n + m));
    constraint_val_.emplace_back(Vector::Zero(p));
    constraint_dual_.emplace_back(Vector::Zero(p));
  }

  // Backward pass data
  lxx_ = Matrix::Zero(n, n);
  luu_ = Matrix::Zero(m, m);
  lux_ = Matrix::Zero(m, n);
  lx_ = Vector::Zero(n);
  lu_ = Vector::Zero(m);

  if (!DynamicsAreLinear()) {
    A_ = Matrix::Zero(n2, n);
    B_ = Matrix::Zero(n2, m);
    f_ = Vector::Zero(n2);
  }

  Qxx_ = Matrix::Zero(n, n);
  Quu_ = Matrix::Zero(m, m);
  Qux_ = Matrix::Zero(m, n);
  Qx_ = Vector::Zero(n);
  Qu_ = Vector::Zero(m);

  Quu_fact = Quu_.llt();
  K_ = Matrix::Zero(m, n);
  d_ = Vector::Zero(m);

  P_ = Matrix::Zero(n, n);
  p_ = Vector::Zero(n);

  // Calculate Hessian if it's constant
  // Note: it's only truly constant if it's unconstrained
  if (CostFunctionIsQuadratic()) {
    CalcOriginalCostHessian();
  }

  // If data is linear-quadratic, assume we're solving a TVLQR problem
  // Set the gradient equal to the linear cost
  if (IsTerminalKnotPoint() && CostFunctionIsQuadratic()) {
    lx_ = q_;
  }
  if (!IsTerminalKnotPoint() && CostFunctionIsQuadratic() && DynamicsAreLinear()) {
    lx_ = q_;
    lu_ = r_;
  }

  is_initialized_ = true;
  return ErrorCodes::NoError;
}

/////////////////////////////////////////////
// Computational Methods
/////////////////////////////////////////////

ErrorCodes KnotPointData::CalcCostExpansion(bool force_update) {
  bool cost_fun_is_quadratic = cost_fun_type_ != CostFunType::Generic;
  bool is_constrained = NumConstraints() > 0;
  bool must_calc_hessian = !cost_fun_is_quadratic || is_constrained;

  // Calculate expansion of original cost function
  // NOTE: these override the contents in lxx_, luu_, lux_, lx_, lu_
  CalcOriginalCostGradient();
  if (force_update || must_calc_hessian) {
    CalcOriginalCostHessian();
  }

  // TODO: Calculate constraint costs
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::CalcDynamicsExpansion() {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  if (!DynamicsAreLinear()) {
    int n = GetStateDim();
    int m = GetInputDim();
    float h = GetTimeStep();
    dynamics_jacobian_(dynamics_jac_.data(), x_.data(), u_.data(), h);
    A_ = dynamics_jac_.leftCols(n);
    B_ = dynamics_jac_.rightCols(m);
  }
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::CalcActionValueExpansion(const altro::KnotPointData &next) {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  // TODO: allocate storage for the temporary variables here
  Qxx_ = lxx_ + A_.transpose() * next.P_ * A_;
  Quu_ = luu_ + B_.transpose() * next.P_ * B_;
  Qux_ = lux_ + B_.transpose() * next.P_ * A_;
  Qx_ = lx_ + A_.transpose() * (next.P_ * f_ + next.p_);
  Qu_ = lu_ + B_.transpose() * (next.P_ * f_ + next.p_);
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::CalcGains() {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  Quu_fact.compute(Quu_);
  // TODO: combine these solves into 1
  K_ = Qux_;
  d_ = Qu_;
  d_ *= -1;
  Quu_fact.solveInPlace(K_);
  Quu_fact.solveInPlace(d_);
  if (Quu_fact.info() != Eigen::Success) {
    return ErrorCodes::CholeskyFailed;
  }
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::CalcCostToGo() {
  if (IsTerminalKnotPoint()) return ErrorCodes::InvalidOpAtTeriminalKnotPoint;
  P_ = Qxx_ + K_.transpose() * Quu_ * K_ - K_.transpose() * Qux_ - Qux_.transpose() * K_;
  p_ = Qx_ - K_.transpose() * Quu_ * d_ - K_.transpose() * Qu_ + Qux_.transpose() * d_;
  P_ = 0.5 * (P_ + P_.transpose());
  delta_V_[0] = d_.dot(Qu_);
  delta_V_[1] = 0.5 * d_.dot(Quu_ * d_);
  return ErrorCodes::NoError;
}

ErrorCodes KnotPointData::CalcTerminalCostToGo() {
  if (!IsTerminalKnotPoint()) return ErrorCodes::OpOnlyValidAtTerminalKnotPoint;
  P_ = lxx_;
  p_ = lx_;
  return ErrorCodes::NoError;
}


a_float KnotPointData::CalcOriginalCost() {
  a_float J;
  int n = GetStateDim();
  int m = GetInputDim();
  switch (cost_fun_type_) {
    case CostFunType::Generic: {
      J = cost_function_(x_.data(), u_.data());
      break;
    }
    case CostFunType::Quadratic: {
      J = 0.5 * x_.transpose() * Q_.reshaped(n, n) * x_;
      J += q_.dot(x_);
      if (!IsTerminalKnotPoint()) {
        J += 0.5 * u_.transpose() * R_.reshaped(m, m) * u_;
        J += r_.dot(u_);
        J += u_.dot(H_ * x_);
      }
      J += c_;
      break;
    }
    case CostFunType::Diagonal: {
      J = 0.5 * x_.transpose() * Q_.head(n).asDiagonal() * x_;
      J += q_.dot(x_);
      if (!IsTerminalKnotPoint()) {
        J += 0.5 * u_.transpose() * R_.head(m).asDiagonal() * u_;
        J += r_.dot(u_);
      }
      J += c_;
      break;
    }
  }
  return J;
}

void KnotPointData::CalcOriginalCostGradient() {
  int n = GetStateDim();
  int m = GetInputDim();
  switch (cost_fun_type_) {
    case CostFunType::Generic: {
      cost_gradient_(lx_.data(), lu_.data(), x_.data(), u_.data());
      break;
    }
    case CostFunType::Quadratic: {
      lx_ = Q_.reshaped(n, n) * x_;
      lx_ += q_;

      if (!IsTerminalKnotPoint()) {
        lu_ = R_.reshaped(m, m) * u_;
        lu_ += r_;
        lu_ += H_ * x_;
        lx_ += H_.transpose() * u_;
      }
      break;
    }
    case CostFunType::Diagonal: {
      lx_ = Q_.head(n).asDiagonal() * x_;
      lx_ += q_;

      if (!IsTerminalKnotPoint()) {
        lu_ = R_.head(m).asDiagonal() * u_;
        lu_ += r_;
      }
      break;
    }
  }
}

void KnotPointData::CalcOriginalCostHessian() {
  int n = GetStateDim();
  int m = GetInputDim();
  switch (cost_fun_type_) {
    case CostFunType::Generic: {
      cost_hessian_(lxx_.data(), luu_.data(), lux_.data(), x_.data(), u_.data());
      break;
    }
    case CostFunType::Quadratic: {
      lxx_ = Q_.reshaped(n, n);
      luu_ = R_.reshaped(m, m);
      lux_ = H_;
      break;
    }
    case CostFunType::Diagonal: {
      lxx_ = Q_.head(n).asDiagonal();
      luu_ = R_.head(m).asDiagonal();
      lux_.setZero();
      break;
    }
  }
}


}  // namespace altro