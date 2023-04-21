//
// Created by Brian Jackson on 9/28/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

enum CubicSplineReturnCodes {
  CS_NOERROR,
  CS_FOUND_MINIMUM,
  CS_INVALIDPOINTER,
  CS_SADDLEPOINT,
  CS_NOMINIMUM,
  CS_IS_POSITIVE_QUADRATIC,
  CS_IS_LINEAR,
  CS_IS_CONSTANT,
  CS_UNEXPECTED_ERROR,
  CS_SAME_POINT,
};

void CubicSpline_PrintReturnCode(enum CubicSplineReturnCodes err);

typedef struct CubicSpline {
  double x0;
  double a;
  double b;
  double c;
  double d;
} CubicSpline;

// clang-format off
CubicSpline CubicSpline_From2Points(double x1, double y1, double d1,
                                    double x2, double y2, double d2,
                                    enum CubicSplineReturnCodes* err);

CubicSpline CubicSpline_From3Points(double x0, double y0, double d0,
                                    double x1, double y1,
                                    double x2, double y2,
                                    enum CubicSplineReturnCodes* err);

CubicSpline QuadraticSpline_From2Points(double x0, double y0, double d0,
                                        double x1, double y1,
                                        enum CubicSplineReturnCodes* err);
// clang-format on
int CubicSpline_IsValid(const CubicSpline* p);

double CubicSpline_Eval(const CubicSpline* p, double x, enum CubicSplineReturnCodes* err);

double CubicSpline_ArgMin(const CubicSpline* p, enum CubicSplineReturnCodes* err);

int CubicSpline_IsQuadratic(const CubicSpline* p, enum CubicSplineReturnCodes* err);
