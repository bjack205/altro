//
// Created by Brian Jackson on 9/28/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "cubicspline.h"
#include "stdio.h"
#include "math.h"

#define LINESEARCH_TOL 1e-6

#define CHECK_VALID_ERROR_POINTER(err) \
  enum CubicSplineReturnCodes err0;  \
  if (!err) err = &err0;

enum CubicSplineReturnCodes QuadraticFormula(double *x1, double *x2, double a, double b, double c);

CubicSpline CubicSpline_From2Points(double x1, double y1, double d1, double x2, double y2,
                                    double d2, enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);

  double delta = x2 - x1;
  if (fabs(delta) < LINESEARCH_TOL) {
    *err = CS_SAME_POINT;
    CubicSpline result = {NAN, NAN, NAN, NAN, NAN};
    return result;
  }

  double a = y1;
  double b = d1;
  double c = 3 * (y2 - y1) / (delta * delta) - (d2 + 2 * d1) / delta;
  double d = (d2 + d1) / (delta * delta) - 2 * (y2 - y1) / (delta * delta * delta);
  CubicSpline result = {
      .x0 = x1,
      .a = a,
      .b = b,
      .c = c,
      .d = d,
  };
  *err = CS_NOERROR;
  return result;
}

CubicSpline CubicSpline_From3Points(double x0, double y0, double d0, double x1, double y1,
                                    double x2, double y2, enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);
  double delta1 = x1 - x0;
  double delta2 = x2 - x0;
  if (fabs(delta1) < LINESEARCH_TOL || fabs(delta2) < LINESEARCH_TOL) {
    *err = CS_SAME_POINT;
    CubicSpline result = {NAN, NAN, NAN, NAN, NAN};
    return result;
  }

  double dy1 = (y1 - y0) / (delta1 * delta1) - d0 / delta1;
  double dy2 = (y2 - y0) / (delta2 * delta2) - d0 / delta2;
  double s = 1/(delta2 - delta1);

  double a = y0;
  double b = d0;
  double c = dy1 * (1 + delta1 * s) - dy2 * delta1 * s;
  double d = -dy1 * s + dy2 * s;
  CubicSpline result = {
      .x0 = x0,
      .a = a,
      .b = b,
      .c = c,
      .d = d,
  };
  return result;
}

CubicSpline QuadraticSpline_From2Points(double x0, double y0, double d0, double x1, double y1,
                                        enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);
  double delta = x1 - x0;
  double dy = (y1 - y0) / (delta * delta) - d0 / delta;

  CubicSpline result = {
      .x0 = x0,
      .a = y0,
      .b = d0,
      .c = dy,
      .d = 0,
  };
  *err = CS_NOERROR;
  return result;
}

int CubicSpline_IsValid(const CubicSpline *p) {
  if (!p) return 0;
  if (isfinite(p->x0) && isfinite(p->a) && isfinite(p->b) && isfinite(p->c) && isfinite(p->d)) {
    return 1;
  }
  return 0;
}


double CubicSpline_Eval(const CubicSpline *p, double x, enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);
  if (!p) {
    *err = CS_INVALIDPOINTER;
    return NAN;
  }
  *err = CS_NOERROR;
  double delta = x - p->x0;
  double y = p->a + p->b * delta + p->c * delta * delta + p->d * delta * delta * delta;
  return y;
}

double CubicSpline_ArgMin(const CubicSpline *p, enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);

  // Check if spline pointer is valid
  if (!p) {
    *err = CS_INVALIDPOINTER;
    return NAN;
  }

  // Extract coefficients
  double b = p->b;
  double c = p->c;
  double d = p->d;

  // Check if it's quadratic
  int is_quadratic = fabs(d) < LINESEARCH_TOL;
  int is_linear = is_quadratic && (fabs(c) < LINESEARCH_TOL);
  int is_constant = is_linear && (fabs(b) < LINESEARCH_TOL);
  if (is_quadratic) {
    if (is_linear) {
      if (is_constant) {
        *err = CS_IS_CONSTANT;
      } else {
        *err = CS_IS_LINEAR;
      }
      return NAN;
    } else if (c <= 0) {
      *err = CS_IS_POSITIVE_QUADRATIC;
      return NAN;
    } else {
      *err = CS_FOUND_MINIMUM;
      return -b / (2 * c) + p->x0;
    }
  }

  // Check stationary points
  double d1, d2;
  enum CubicSplineReturnCodes err_qform = QuadraticFormula(&d1, &d2, 3 * d, 2 * c, b);
  if (err_qform != CS_NOERROR) {
    *err = err_qform;
    return NAN;
  }

  // Check the curvature at these points
  double curv1 = 2 * c + 6 * d * d1;
  double curv2 = 2 * c + 6 * d * d2;

  // Pick the one that's positive
  // If both are positive or negative, return a NAN
  double x1 = d1 + p->x0;
  double x2 = d2 + p->x0;
  if (fabs(curv1) < LINESEARCH_TOL && fabs(curv2) < LINESEARCH_TOL) {
    // Occurs when roots are repeated
    *err = CS_SADDLEPOINT;
    return NAN;;
  } else if (curv1 > 0 && curv2 < 0) {
    *err = CS_FOUND_MINIMUM;
    return x1;
  } else if (curv1 < 0 && curv2 > 0) {
    *err = CS_FOUND_MINIMUM;
    return x2;
  } else {
    // roots of a quadratic are:
    // 1. both imaginary
    // 2. both real (with opposite curvature)
    // 3. repeated (with zero curvature)
    // So if it reaches here, something wierd happened
    *err = CS_UNEXPECTED_ERROR;
    return NAN;
  }
}

int CubicSpline_IsQuadratic(const CubicSpline *p, enum CubicSplineReturnCodes *err) {
  CHECK_VALID_ERROR_POINTER(err);
  if (!p) {
    *err = CS_INVALIDPOINTER;
    return 0;
  }
  *err = CS_NOERROR;
  return fabs(p->d) < LINESEARCH_TOL;
}

void CubicSpline_PrintReturnCode(enum CubicSplineReturnCodes err) {
  switch (err) {
    case CS_NOERROR:
      puts("No error");
      break;
    case CS_FOUND_MINIMUM:
      puts("Minimum successfully found");
      break;
    case CS_INVALIDPOINTER:
      puts("Invalid pointer");
      break;
    case CS_SADDLEPOINT:
      puts("No minimum. Is a saddle point");
      break;
    case CS_NOMINIMUM:
      break;
    case CS_IS_POSITIVE_QUADRATIC:
      puts("No minimum. Is a positive quadratic");
      break;
    case CS_IS_LINEAR:
      puts("No minimum. Is linear");
      break;
    case CS_UNEXPECTED_ERROR:
      puts("Unexpected error");
      break;
    case CS_IS_CONSTANT:
      puts("No unique minimum. Is constant");
      break;
    case CS_SAME_POINT:
      puts("Spline constructor given the same x coordinate. Must be separate points");
      break;
    default:
      break;
  }
}

enum CubicSplineReturnCodes QuadraticFormula(double *x1, double *x2, double a, double b, double c) {
  if (!x1) return CS_INVALIDPOINTER;
  if (!x2) return CS_INVALIDPOINTER;
  if (fabs(a) < LINESEARCH_TOL) return CS_IS_LINEAR;

  double s2 = b * b - 4 * a * c;
  double s;
  if (fabs(s2) < LINESEARCH_TOL) {
    s = 0.0;
  } else if (s2 < 0) {
    return CS_NOMINIMUM;
  } else {
    s = sqrt(s2);
  }
  *x1 = (-b + s) / (2 * a);
  *x2 = (-b - s) / (2 * a);
  return CS_NOERROR;
}
