#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

extern "C" {
#include "linesearch/cubicspline.h"
}

TEST(CubicSplineTests, EvalConstant) {
  const double a = 1.2;
  CubicSpline constant_spline = {.x0 = 0, .a = a, .b = 0.0, .c = 0.0, .d = 0.0};
  CubicSpline *p = &constant_spline;
  CubicSplineReturnCodes err;

  std::vector<double> test_x = {-1, 0, 1, 2};
  for (double x : test_x) {
    double y = CubicSpline_Eval(p, 0.0, &err);
    EXPECT_EQ(err, CS_NOERROR);
    EXPECT_DOUBLE_EQ(y, a);
  }
}

TEST(CubicSplineTests, ArgMin_Constant) {
  const double a = 1.2;
  CubicSplineReturnCodes err;
  CubicSpline constant_spline = CubicSpline_From2Points(0, a, 0, 1, a, 0, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);
  EXPECT_DOUBLE_EQ(p->a, a);
  EXPECT_DOUBLE_EQ(p->b, 0.0);
  EXPECT_DOUBLE_EQ(p->c, 0.0);
  EXPECT_DOUBLE_EQ(p->d, 0.0);

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_IS_CONSTANT);
  EXPECT_TRUE(std::isnan(x_min));
}

TEST(CubicSplineTests, ArgMin_Linear) {
  const double b = 1;  // slope
  CubicSplineReturnCodes err;
  CubicSpline constant_spline = CubicSpline_From2Points(0, 0, b, 1, b, b, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);
  EXPECT_DOUBLE_EQ(p->a, 0.0);
  EXPECT_DOUBLE_EQ(p->b, b);
  EXPECT_DOUBLE_EQ(p->c, 0.0);
  EXPECT_DOUBLE_EQ(p->d, 0.0);

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_IS_LINEAR);
  EXPECT_TRUE(std::isnan(x_min));
}

TEST(CubicSplineTests, ArgMin_PositiveQuadratic) {
  const double center = 0.5;
  const double off = 0.2;
  const double slope = 1.0;
  CubicSplineReturnCodes err;
  CubicSpline constant_spline =
      CubicSpline_From2Points(center - off, 0, -slope, center + off, 0, slope, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);
  EXPECT_DOUBLE_EQ(p->d, 0);
  EXPECT_TRUE(CubicSpline_IsQuadratic(p, &err));
  EXPECT_EQ(err, CS_NOERROR);

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_FOUND_MINIMUM);
  EXPECT_DOUBLE_EQ(x_min, center);

}

TEST(CubicSplineTests, ArgMin_NegativeQuadratic) {
  const double center = 0.5;
  const double off = 0.2;
  const double slope = 1.0;
  CubicSplineReturnCodes err;
  CubicSpline constant_spline =
      CubicSpline_From2Points(center - off, 0, +slope, center + off, 0, -slope, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);
  EXPECT_DOUBLE_EQ(p->d, 0);
  EXPECT_TRUE(CubicSpline_IsQuadratic(p, &err));
  EXPECT_EQ(err, CS_NOERROR);

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_IS_POSITIVE_QUADRATIC);
  EXPECT_TRUE(std::isnan(x_min));
}

TEST(CubicSplineTests, ArgMin_Cubic) {
  CubicSplineReturnCodes err;
  CubicSpline constant_spline =
      CubicSpline_From2Points(0,0,-1, 1,0,2, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);
  EXPECT_DOUBLE_EQ(p->d, 1.0);

  double x_min = CubicSpline_ArgMin(p, &err);
  double x_min_expected = 0.5773502691896257;
  EXPECT_EQ(err, CS_FOUND_MINIMUM);
  EXPECT_LT(std::abs(x_min - x_min_expected), 1e-10);
}

TEST(CubicSplineTests, ArgMin_CubicNoMin) {
  CubicSplineReturnCodes err;
  CubicSpline constant_spline =
      CubicSpline_From2Points(0,0,-1, 1,-3,-10, &err);
  CubicSpline *p = &constant_spline;
  EXPECT_EQ(err, CS_NOERROR);

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_NOMINIMUM);
  EXPECT_TRUE(std::isnan(x_min));
}

TEST(CubicSplineTests, ArgMin_CubicSaddlePoint) {
  CubicSplineReturnCodes err;
  CubicSpline constant_spline = {0, 0, 1.0, -1.0, 1.0/3.0};
  CubicSpline *p = &constant_spline;

  double x_min = CubicSpline_ArgMin(p, &err);
  EXPECT_EQ(err, CS_SADDLEPOINT);
  EXPECT_TRUE(std::isnan(x_min));
  EXPECT_FALSE(CubicSpline_IsQuadratic(p, &err));
  EXPECT_EQ(err, CS_NOERROR);
}
