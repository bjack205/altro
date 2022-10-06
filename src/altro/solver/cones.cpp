//
// Created by Brian Jackson on 10/6/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "cones.hpp"

#include <Eigen/Dense>
using Eigen::MatrixXd;

namespace altro {

void SecondOrderConeProjection(int dim, const a_float *x, a_float *px) {
  // assumes that x is stacked [v; s] such that ||v|| <= s
  int n = dim - 1;
  a_float s = x[n];
  a_float a = 0.0;
  for (int i = 0; i < n; ++i) {
    a += x[i] * x[i];
  }

  a = std::sqrt(a);
  if (a <= -s) {  // below the cone
    for (int i = 0; i < dim; ++i) {
      px[i] = 0.0;
    }
  } else if (a <= s) {  // in the cone
    for (int i = 0; i < dim; ++i) {
      px[i] = x[i];
    }
  } else {  // outside the cone
    a_float c = 0.5 * (1 + s / a);
    for (int i = 0; i < n; ++i) {
      px[i] = c * x[i];
    }
    px[n] = c * a;
  }
}

void SecondOrderConeJacobian(int dim, const a_float *x, a_float *jac) {
  int n = dim - 1;
  a_float s = x[n];
  a_float a = 0.0;
  for (int i = 0; i < n; ++i) {
    a += x[i] * x[i];
  }
  a = std::sqrt(a);

  Eigen::Map<MatrixXd> J(jac, dim, dim);
  if (a <= -s) {
    J.setZero();
  } else if (a <= s) {
    J.setIdentity();
  } else {
    // scalar
    a_float c = 0.5 * (1 + s / a);

    // dvdv
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        J(i, j) = -0.5 * s / (a * a * a) * x[i] * x[j];
        J(i, j) += (i == j) ? c : 0;
      }
    }

    // dvds
    for (int i = 0; i < n; ++i) {
      J(i, n) = 0.5 * x[i] / a;
    }

    // ds
    for (int j = 0; j < n; ++j) {
      J(n, j) = ((-0.5 * s / (a * a)) + c / a) * x[j];
    }
    J(n, n) = 0.5;
  }
}

void SecondOrderConeHessian(int dim, const a_float *x, const a_float *b, a_float *hess) {
  int n = dim - 1;
  a_float s = x[n];
  a_float bs = b[n];
  a_float vbv = 0;  // dot product of vector parts of x and b
  a_float a = 0;
  for (int i = 0; i < n; ++i) {
    a += x[i] * x[i];
    vbv += x[i] * b[i];
  }
  a = std::sqrt(a);

  Eigen::Map<MatrixXd> H(hess, dim, dim);
  if (a <= -s) {
    H.setZero();
  } else if (a <= s) {
    H.setZero();
  } else {
    for (int i = 0; i < n; ++i) {
      a_float hi = 0;
      for (int j = 0; j < n; ++j) {
        a_float Hij = -x[i] * x[j] / (a * a);
        Hij += (i == j) ? 1 : 0;
        hi += Hij * b[j];
      }
      H(i, n) = hi / (2 * a);
      H(n, i) = hi / (2 * a);
      for (int j = 0; j <= i; ++j) {
        a_float vij = x[i] * x[j];
        a_float H1 = hi * x[j] * (-s / (a * a * a));
        a_float H2 = vij * (2 * vbv) / (a * a * a * a) - x[i] * b[j] / (a * a);
        a_float H3 = -vij / (a * a);
        if (i == j) {
          H2 -= vbv / (a * a);
          H3 += 1;
        }
        H2 *= s / a;
        H3 *= bs / a;
        H(i, j) = (H1 + H2 + H3) / 2.0;
        H(j, i) = (H1 + H2 + H3) / 2.0;
      }
    }
    H(n, n) = 0.0;
  }
}

ErrorCodes ConicProjection(ConstraintType cone, int dim, const a_float *x, a_float *px) {
  if (!x) return ALTRO_THROW("Bad input pointer in ConicProjection", ErrorCodes::InvalidPointer);
  if (!px) return ALTRO_THROW("Bad output pointer in ConicProjection", ErrorCodes::InvalidPointer);

  switch (cone) {
    case ConstraintType::EQUALITY:  // Zero cone
      for (int i = 0; i < dim; ++i) {
        px[i] = 0;
      }
      break;
    case ConstraintType::IDENTITY:
      for (int i = 0; i < dim; ++i) {
        px[i] = x[i];
      }
      break;
    case ConstraintType::INEQUALITY:  // negative orthant
      for (int i = 0; i < dim; ++i) {
        px[i] = std::min(0.0, x[i]);
      }
      break;
    case ConstraintType::SECOND_ORDER_CONE:
      SecondOrderConeProjection(dim, x, px);
      break;
  }
  return ErrorCodes::NoError;
}

ErrorCodes ConicProjectionJacobian(ConstraintType cone, int dim, const a_float *x, a_float *jac) {
  if (!x)
    return ALTRO_THROW("Bad input pointer in ConicProjectionJacobian", ErrorCodes::InvalidPointer);
  if (!jac)
    return ALTRO_THROW("Bad output pointer in ConicProjectionJacobian", ErrorCodes::InvalidPointer);

  Eigen::Map<MatrixXd> J(jac, dim, dim);
  switch (cone) {
    case ConstraintType::EQUALITY:
      J.setZero();
      break;
    case ConstraintType::IDENTITY:
      J.setIdentity();
      break;
    case ConstraintType::INEQUALITY:
      J.setZero();
      for (int i = 0; i < dim; ++i) {
        J(i, i) = (x[i] <= 0) ? 1 : 0;
      }
      break;
    case ConstraintType::SECOND_ORDER_CONE:
      SecondOrderConeJacobian(dim, x, jac);
      break;
  }
  return ErrorCodes::NoError;
}

ErrorCodes ConicProjectionHessian(ConstraintType cone, int dim, const a_float *x, const a_float *b,
                                  a_float *hess) {
  if (!x)
    return ALTRO_THROW("Bad input pointer in ConicProjectionHessian", ErrorCodes::InvalidPointer);
  if (!hess)
    return ALTRO_THROW("Bad output pointer in ConicProjectionHessian", ErrorCodes::InvalidPointer);

  Eigen::Map<MatrixXd> H(hess, dim, dim);
  switch (cone) {
    case ConstraintType::EQUALITY:
      H.setZero();
      break;
    case ConstraintType::IDENTITY:
      H.setZero();
      break;
    case ConstraintType::INEQUALITY:
      H.setZero();
      break;
    case ConstraintType::SECOND_ORDER_CONE:
      SecondOrderConeHessian(dim, x, b, hess);
      break;
  }
  return ErrorCodes::NoError;
}

}  // namespace altro
