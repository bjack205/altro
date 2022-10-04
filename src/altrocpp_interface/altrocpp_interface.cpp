//
// Created by Brian Jackson on 9/13/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "altrocpp_interface.hpp"

namespace altro {
namespace cpp_interface {

GeneralDiscreteDynamics::GeneralDiscreteDynamics(int n, int m,
                                                 ExplicitDynamicsFunction dynamics_function,
                                                 ExplicitDynamicsJacobian dynamics_jacobian)
    : num_states_(n),
      num_inputs_(m),
      dynamics_function_(dynamics_function),
      dynamics_jacobian_(dynamics_jacobian) {}

void GeneralDiscreteDynamics::Evaluate(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                                       Eigen::Ref<VectorXd> xnext) {
  (void)t;
  dynamics_function_(xnext.data(), x.data(), u.data(), h);
}

void GeneralDiscreteDynamics::Jacobian(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                                       Eigen::Ref<MatrixXd> jac) {
  (void)t;
  dynamics_jacobian_(jac.data(), x.data(), u.data(), h);
}

void GeneralDiscreteDynamics::Hessian(const VectorXdRef& x, const VectorXdRef& u, float t, float h,
                                      const VectorXdRef& b, Eigen::Ref<MatrixXd> hess) {
  (void)x;
  (void)u;
  (void)t;
  (void)h;
  (void)b;
  (void)hess;
}

double QuadraticCost::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  return 0.5 * x.dot(Q_ * x) + x.dot(H_ * u) + 0.5 * u.dot(R_ * u) + q_.dot(x) + r_.dot(u) + c_;
}

void QuadraticCost::Gradient(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<VectorXd> dx,
                             Eigen::Ref<VectorXd> du) {
  dx = Q_ * x + q_ + H_ * u;
  du = R_ * u + r_ + H_.transpose() * x;
}

void QuadraticCost::Hessian(const VectorXdRef& x, const VectorXdRef& u, Eigen::Ref<MatrixXd> dxdx,
                            Eigen::Ref<MatrixXd> dxdu, Eigen::Ref<MatrixXd> dudu) {
  ALTRO_UNUSED(x);
  ALTRO_UNUSED(u);
  dxdx = Q_;
  dudu = R_;
  dxdu = H_;
}

void QuadraticCost::Validate() {
  ALTRO_ASSERT(Q_.rows() == n_, "Q has the wrong number of rows");
  ALTRO_ASSERT(Q_.cols() == n_, "Q has the wrong number of columns");
  ALTRO_ASSERT(R_.rows() == m_, "R has the wrong number of rows");
  ALTRO_ASSERT(R_.cols() == m_, "R has the wrong number of columns");
  ALTRO_ASSERT(H_.rows() == n_, "H has the wrong number of rows");
  ALTRO_ASSERT(H_.cols() == m_, "H has the wrong number of columns");

  // Check symmetry of Q and R
  ALTRO_ASSERT(Q_.isApprox(Q_.transpose()), "Q is not symmetric");
  ALTRO_ASSERT(R_.isApprox(R_.transpose()), "R is not symmetric");

  // Check that R is positive definite
  if (!terminal_) {
    Rfact_.compute(R_);
    ALTRO_ASSERT(Rfact_.info() == Eigen::Success, "R must be positive definite");
  }

  // Check if Q is positive semidefinite
  Qfact_.compute(Q_);
  ALTRO_ASSERT(Qfact_.info() == Eigen::Success,
               "The LDLT decomposition could of Q could not be computed. "
               "Must be positive semi-definite");
  Eigen::Diagonal<const MatrixXd> D = Qfact_.vectorD();
  bool ispossemidef = true;
  (void)ispossemidef;  // surpress erroneous unused variable error
  for (int i = 0; i < n_; ++i) {
    if (D(i) < 0) {
      ispossemidef = false;
      break;
    }
  }
  ALTRO_ASSERT(ispossemidef, "Q must be positive semi-definite");
}

GeneralCostFunction::GeneralCostFunction(int n, int m, altro::CostFunction cost_function,
                                         CostGradient cost_gradient, CostHessian cost_hessian)
    : num_states_(n),
      num_inputs_(m),
      cost_function_(cost_function),
      cost_gradient_(cost_gradient),
      cost_hessian_(cost_hessian) {}

double GeneralCostFunction::Evaluate(const VectorXdRef& x, const VectorXdRef& u) {
  a_float J = cost_function_(x.data(), u.data());
  return static_cast<double>(J);
}

void GeneralCostFunction::Gradient(const VectorXdRef& x, const VectorXdRef& u,
                                   Eigen::Ref<VectorXd> dx, Eigen::Ref<VectorXd> du) {
  cost_gradient_(dx.data(), du.data(), x.data(), u.data());
}

void GeneralCostFunction::Hessian(const VectorXdRef& x, const VectorXdRef& u,
                                  Eigen::Ref<MatrixXd> dxdx, Eigen::Ref<MatrixXd> dudx,
                                  Eigen::Ref<MatrixXd> dudu) {
  cost_hessian_(dxdx.data(), dudu.data(), dudx.data(), x.data(), u.data());
}

}  // namespace cpp_interface
}  // namespace altro
