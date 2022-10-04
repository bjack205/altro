//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "exceptions.hpp"
#include "internal_types.hpp"

namespace altro {

class KnotPointData {
  static constexpr int kMaxConstraints = std::numeric_limits<int>::max();
  enum class CostFunType { Generic, Quadratic, Diagonal };

 public:
  explicit KnotPointData(bool is_terminal);

  // Prohibit copying
  KnotPointData(const KnotPointData& data) = delete;
  KnotPointData operator=(const KnotPointData& data) = delete;

  // Allow moving
  KnotPointData(KnotPointData&& other) = default;

  // Setters
  ErrorCodes SetDimension(int num_states, int num_inputs);
  ErrorCodes SetNextStateDimension(int num_states_next);
  ErrorCodes SetTimestep(float h);

  ErrorCodes SetQuadraticCost(int n, int m, const a_float *Qmat, const a_float *Rmat,
                              const a_float *Hmat, const a_float *q, const a_float *r, a_float c);
  ErrorCodes SetDiagonalCost(int n, int m, const a_float *Qdiag, const a_float *Rdiag,
                             const a_float *q, const a_float *r, a_float c);
  ErrorCodes SetCostFunction(CostFunction cost_function, CostGradient cost_gradient,
                             CostHessian cost_hessian);

  ErrorCodes SetLinearDynamics(int n2, int n, int m, const a_float *A, const a_float *B,
                               const a_float *f = nullptr);
  ErrorCodes SetDynamics(ExplicitDynamicsFunction dynamics_function,
                         ExplicitDynamicsJacobian dynamics_jacobian);

  ErrorCodes SetConstraint(ConstraintFunction constraint_function,
                           ConstraintJacobian constraint_jacobian, int dim,
                           ConstraintType constraint_type, std::string label);

  ErrorCodes Initialize();

  // Calculation methods
  a_float CalcCost();
  ErrorCodes CalcCostGradient();
  ErrorCodes CalcCostExpansion(bool force_update);
  ErrorCodes CalcDynamics(a_float *xnext);
  ErrorCodes CalcDynamicsExpansion();
  ErrorCodes CalcActionValueExpansion(const KnotPointData &next);
  ErrorCodes CalcGains();
  ErrorCodes CalcCostToGo();
  ErrorCodes CalcTerminalCostToGo();

  // Getters
  int GetNextStateDim() const { return num_next_state_; }

  int GetStateDim() const { return num_states_; }

  int GetInputDim() const { return num_inputs_; }

  float GetTimeStep() const { return h_; }

  bool IsInitialized() const { return is_initialized_; }

  bool IsTerminalKnotPoint() const { return is_terminal_; }

  int NumConstraints() const { return static_cast<int>(constraint_dims_.size()); }

  bool DynamicsAreLinear() const { return dynamics_are_linear_; };

  bool CostFunctionIsQuadratic() const { return cost_fun_type_ != CostFunType::Generic; }

 private:
  /////////////////////////////////////////////
  // Definition
  /////////////////////////////////////////////

  // General info
  int num_next_state_ = 0;
  int num_states_ = 0;
  int num_inputs_ = 0;
  float h_ = 0.0;

  // Cost function
  CostFunType cost_fun_type_;
  CostFunction cost_function_;
  CostGradient cost_gradient_;
  CostHessian cost_hessian_;

  Vector Q_;
  Vector R_;
  Matrix H_;
  Vector q_;
  Vector r_;
  a_float c_;

  // Dynamics
  bool dynamics_are_linear_ = false;
  Vector affine_term_;
  ExplicitDynamicsFunction dynamics_function_;
  ExplicitDynamicsJacobian dynamics_jacobian_;

  // Bound constraint
  Vector x_hi_;
  Vector x_lo_;
  Vector u_hi_;
  Vector u_lo_;

  // Constraints
  std::vector<ConstraintFunction> constraint_function_;
  std::vector<ConstraintJacobian> constraint_jacobian_;
  std::vector<int> constraint_dims_;
  std::vector<ConstraintType> constraint_type_;
  std::vector<std::string> constraint_label_;

  // Flags
  bool dims_are_set_ = false;
  bool cost_fun_is_set_ = false;
  bool dynamics_is_set_ = false;
  bool is_initialized_ = false;
  bool is_terminal_ = false;

 public:  // NOTE: making this data public for now so it's easy to access
  // States and controls
  Vector x;  // state
  Vector u;  // input
  Vector y;  // dynamics dual

  // All functions are calculated using these values
  Vector x_;  // temp state
  Vector u_;  // temp input
  Vector y_;  // temp dynamics dual

  Matrix dynamics_jac_;
  Vector dynamics_val_;
  Vector dynamics_dual_;

  // Bound constraint values
  Vector c_x_hi_;
  Vector c_x_lo_;
  Vector c_u_hi_;
  Vector c_u_lo_;

  // Bound constraint duals
  Vector v_x_hi_;
  Vector v_x_lo_;
  Vector v_u_hi_;
  Vector v_u_lo_;

  std::vector<Matrix> constraint_jac_;
  std::vector<Vector> constraint_val_;
  std::vector<Vector> constraint_dual_;

  // Backward pass
  Matrix lxx_;
  Matrix luu_;
  Matrix lux_;
  Vector lx_;
  Vector lu_;

  Matrix A_;
  Matrix B_;
  Vector f_;

  Matrix Qxx_;
  Matrix Quu_;
  Matrix Qux_;
  Vector Qx_;
  Vector Qu_;

  Matrix Qxx_tmp_;
  Matrix Quu_tmp_;
  Matrix Qux_tmp_;
  Vector Qx_tmp_;
  Vector Qu_tmp_;

  Eigen::LLT<Matrix> Quu_fact;
  Matrix K_;
  Vector d_;  // TODO: make this an extra column of K

  Matrix P_;
  Vector p_;
  a_float delta_V_[2];

  // Forward pass
  Vector dx_da_;   // gradient of x wrt alpha
  Vector du_da_;   // gradient of u wrt alpha

 private:
  a_float CalcOriginalCost();
  void CalcOriginalCostGradient();
  void CalcOriginalCostHessian();
};

}  // namespace altro