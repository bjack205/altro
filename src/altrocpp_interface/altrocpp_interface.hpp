//
// Created by Brian Jackson on 9/13/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <string>

#include "Eigen/Dense"
#include "altro/problem/problem.hpp"
#include "altro/solver/typedefs.hpp"

namespace altro {
namespace cpp_interface {

/**
 * @brief Interface wrapper for AltroCpp that takes generic function pointers.
 */
class GeneralDiscreteDynamics : public problem::DiscreteDynamics {
 public:
  GeneralDiscreteDynamics(int n, int m, ExplicitDynamicsFunction dynamics_function,
                          ExplicitDynamicsJacobian dynamics_jacobian);

  int StateDimension() const override { return num_states_; }

  int ControlDimension() const override { return num_inputs_; }

  int OutputDimension() const override { return num_states_; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                Eigen::Ref<VectorXd> xnext) override;

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                Eigen::Ref<MatrixXd> jac) override;

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h, const VectorXdRef& b,
               Eigen::Ref<MatrixXd> hess) override;

  bool HasHessian() const override { return false; }

 private:
  int num_states_;
  int num_inputs_;
  ExplicitDynamicsFunction dynamics_function_;
  ExplicitDynamicsJacobian dynamics_jacobian_;
};

class GeneralCostFunction : public problem::CostFunction {
 public:
  GeneralCostFunction(int n, int m, altro::CostFunction cost_function, CostGradient cost_gradient,
                      CostHessian cost_hessian);

  int StateDimension() const override { return num_states_; }

  int ControlDimension() const override { return num_inputs_; }

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;

  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override;

  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
               Eigen::Ref<MatrixXd> dudx, Eigen::Ref<MatrixXd> dudu) override;

 private:
  int num_states_;
  int num_inputs_;
  altro::CostFunction cost_function_;
  CostGradient cost_gradient_;
  CostHessian cost_hessian_;
};

class QuadraticCost : public problem::CostFunction {
 public:
  QuadraticCost(const MatrixXd& Q, const MatrixXd& R, const MatrixXd& H, const VectorXd& q,
                const VectorXd& r, double c = 0, bool terminal = false)
      : n_(q.size()),
        m_(r.size()),
        isblockdiag_(H.norm() < 1e-8),
        Q_(Q),
        R_(R),
        H_(H),
        q_(q),
        r_(r),
        c_(c),
        terminal_(terminal) {
    Validate();
  }

  static QuadraticCost LQRCost(const MatrixXd& Q, const MatrixXd& R, const VectorXd& xref,
                               const VectorXd& uref, bool terminal = false) {
    int n = Q.rows();
    int m = R.rows();
    ALTRO_ASSERT(xref.size() == n, "xref is the wrong size.");
    MatrixXd H = MatrixXd::Zero(n, m);
    VectorXd q = -(Q * xref);
    VectorXd r = -(R * uref);
    double c = 0.5 * xref.dot(Q * xref) + 0.5 * uref.dot(R * uref);
    return QuadraticCost(Q, R, H, q, r, c, terminal);
  }

  int StateDimension() const override { return n_; }

  int ControlDimension() const override { return m_; }

  double Evaluate(const VectorXdRef& x, const VectorXdRef& u) override;
  void Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                Eigen::Ref<VectorXd> du) override;
  void Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
               Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) override;

  const MatrixXd& GetQ() const { return Q_; }

  const MatrixXd& GetR() const { return R_; }

  const MatrixXd& GetH() const { return H_; }

  const VectorXd& Getq() const { return q_; }

  const VectorXd& Getr() const { return r_; }

  double GetConstant() const { return c_; }

  const Eigen::LDLT<MatrixXd>& GetQfact() const { return Qfact_; }

  const Eigen::LLT<MatrixXd>& GetRfact() const { return Rfact_; }

  bool IsBlockDiagonal() const { return isblockdiag_; }

 private:
  void Validate();

  int n_;
  int m_;
  bool isblockdiag_;
  MatrixXd Q_;
  MatrixXd R_;
  MatrixXd H_;
  VectorXd q_;
  VectorXd r_;
  double c_;
  bool terminal_;

  // decompositions of Q and R
  Eigen::LDLT<MatrixXd> Qfact_;
  Eigen::LLT<MatrixXd> Rfact_;
};

template <class Cone>
class GeneralConstraint : public constraints::Constraint<Cone> {
 public:
  explicit GeneralConstraint(int n, int m, int p, ConstraintFunction con, ConstraintJacobian jac,
                             std::string label = "")
      : con_(con), jac_(jac), state_dim_(n), input_dim_(m), constraint_dim_(p), label_(label) {}

  std::string GetLabel() const override { return label_; }
  int StateDimension() const override { return state_dim_; }
  int ControlDimension() const override { return input_dim_; }
  int OutputDimension() const override { return constraint_dim_; }

  void Evaluate(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> c) override {
    con_(c.data(), x.data(), u.data());
  }

  void Jacobian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> jac) override {
    jac_(jac.data(), x.data(), u.data());
  }

  void SetLabel(std::string label) { label_ = std::move(label); }

 private:
  ConstraintFunction con_;
  ConstraintJacobian jac_;
  int state_dim_;
  int input_dim_;
  int constraint_dim_;
  std::string label_;
};

using EqualityConstraint = GeneralConstraint<constraints::Equality>;
using InequalityConstraint = GeneralConstraint<constraints::Inequality>;

}  // namespace cpp_interface
}  // namespace altro