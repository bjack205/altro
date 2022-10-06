#include "altro/solver/cones.hpp"

#include "altro/solver/internal_types.hpp"
#include "altro/utils/formatting.hpp"
#include "gtest/gtest.h"

namespace altro {

TEST(ConicTests, EqualityProjection) {
  int dim = 4;
  Vector x(dim);
  Vector px(dim);
  Vector px_expected(dim);
  x << 0.1, -0.5, 0.2, 0.0;
  px_expected.setZero();
  ConicProjection(ConstraintType::EQUALITY, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);
}

TEST(ConicTests, InqualityProjection) {
  int dim = 4;
  Vector x(dim);
  Vector px(dim);
  Vector px_expected(dim);
  x << 0.1, -0.5, 0.2, 0.0;
  px_expected << 0.0, -0.5, 0.0, 0.0;
  ConicProjection(ConstraintType::INEQUALITY, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);
}

TEST(ConicTests, IdentityProjection) {
  int dim = 4;
  Vector x(dim);
  Vector px(dim);
  Vector px_expected(dim);
  x << 0.1, -0.5, 0.2, 0.0;
  px_expected = x;
  ConicProjection(ConstraintType::IDENTITY, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);
}

TEST(ConicTests, SOCProjection) {
  int dim = 4;
  Vector x(dim);
  Vector px(dim);
  Vector px_expected(dim);
  x << 0.1, -0.5, 0.2, 0.0;
  a_float mag = x.norm();

  // In Cone
  a_float s = mag * 1.1;
  x[dim - 1] = s;
  px_expected = x;
  ConicProjection(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);

  // Below cone
  s = mag * -1.1;
  x[dim - 1] = s;
  px_expected.setZero();
  ConicProjection(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);

  // Outside cone
  s = mag * 0.9;
  x[dim - 1] = s;
  px_expected << 0.095, -0.475, 0.19, 0.5203364296299079;
  ConicProjection(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), px.data());
  EXPECT_LT((px - px_expected).norm(), 1e-10);
}

TEST(ConicTests, EqualityJacobian) {
  int dim = 4;
  Vector x(dim);
  Matrix jac(dim, dim);
  Matrix jac_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  jac_expected.setZero();
  ConicProjectionJacobian(ConstraintType::EQUALITY, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);
}

TEST(ConicTests, IdentityJacobian) {
  int dim = 4;
  Vector x(dim);
  Matrix jac(dim, dim);
  Matrix jac_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  jac_expected.setIdentity();
  ConicProjectionJacobian(ConstraintType::IDENTITY, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);
}

TEST(ConicTests, InequalityJacobian) {
  int dim = 4;
  Vector x(dim);
  Matrix jac(dim, dim);
  Matrix jac_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  jac_expected.diagonal() << 0, 1, 0, 1;
  ConicProjectionJacobian(ConstraintType::INEQUALITY, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);
}

TEST(ConicTests, SOCJacobian) {
  int dim = 4;
  Vector x(dim);
  Matrix jac(dim, dim);
  Matrix jac_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  a_float mag = x.norm();

  // In Cone
  a_float s = mag * 1.1;
  x[dim - 1] = s;
  jac_expected.setIdentity();
  ConicProjectionJacobian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);

  // Below cone
  s = mag * -1.1;
  x[dim - 1] = s;
  jac_expected.setZero();
  ConicProjectionJacobian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);

  // Outside cone
  s = mag * 0.9;
  x[dim - 1] = s;
  jac_expected << 0.9349999999999999, 0.07499999999999998, -0.029999999999999995,
      0.09128709291752768, 0.07499999999999998, 0.5750000000000001, 0.14999999999999997,
      -0.4564354645876384, -0.029999999999999995, 0.14999999999999997, 0.89, 0.18257418583505536,
      0.09128709291752768, -0.45643546458763834, 0.18257418583505536, 0.5;
  ConicProjectionJacobian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), jac.data());
  EXPECT_LT((jac - jac_expected).norm(), 1e-10);
}

TEST(ConicTests, EqualityHessian) {
  int dim = 4;
  Vector x(dim);
  Vector b(dim);
  Matrix hess(dim, dim);
  Matrix hess_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  b << 10, 20, -30, 40;
  hess_expected.setZero();
  ConicProjectionHessian(ConstraintType::EQUALITY, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);
}

TEST(ConicTests, IdentityHessian) {
  int dim = 4;
  Vector x(dim);
  Vector b(dim);
  Matrix hess(dim, dim);
  Matrix hess_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  b << 10, 20, -30, 40;
  hess_expected.setZero();
  ConicProjectionHessian(ConstraintType::IDENTITY, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);
}

TEST(ConicTests, InequalityHessian) {
  int dim = 4;
  Vector x(dim);
  Vector b(dim);
  Matrix hess(dim, dim);
  Matrix hess_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  b << 10, 20, -30, 40;
  hess_expected.setZero();
  ConicProjectionHessian(ConstraintType::INEQUALITY, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);
}

TEST(ConicTests, SOCHessian) {
  int dim = 4;
  Vector x(dim);
  Vector b(dim);
  Matrix hess(dim, dim);
  Matrix hess_expected(dim, dim);
  x << 0.1, -0.5, 0.2, 0.0;
  b << 10, 20, -30, 40;
  a_float mag = x.norm();

  // In Cone
  a_float s = mag * 1.1;
  x[dim - 1] = s;
  hess_expected.setZero();
  ConicProjectionHessian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);

  // Below cone
  s = mag * -1.1;
  x[dim - 1] = s;
  hess_expected.setZero();
  ConicProjectionHessian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);

  // Outside cone
  s = mag * 0.9;
  x[dim - 1] = s;
  hess_expected << 52.54767592811069, 21.83580619450183, -5.434322477800736, 13.69306393762915,
      21.83580619450183, 2.3358061945018775, 6.1716123890036805, -4.564354645876377,
      -5.434322477800736, 6.1716123890036805, 63.146192211409584, -18.257418583505533,
      13.69306393762915, -4.564354645876377, -18.257418583505533, 0.0;
  ConicProjectionHessian(ConstraintType::SECOND_ORDER_CONE, dim, x.data(), b.data(), hess.data());
  EXPECT_LT((hess - hess_expected).norm(), 1e-10);
  EXPECT_LT((hess - hess.transpose()).norm(), 1e-10);  // check symmetry
}

}  // namespace altro
