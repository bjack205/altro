#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

extern "C" {
#include "linesearch/cubicspline.h"
}
#include "linesearch/linesearch.hpp"

TEST(CubicSplineTests, EvalConstant) {
  const double a = 1.2;
  CubicSpline constant_spline = {0, a, 0.0, 0.0, 0.0};
  CubicSpline *p = &constant_spline;
  CubicSplineReturnCodes err;

  std::vector<double> test_x = {-1, 0, 1, 2};
  for (double x : test_x) {
    double y = CubicSpline_Eval(p, x, &err);
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


TEST(LineSearchTests, Quadratic) {
  double c = 1.0;
  double a = 1.0;
  auto phi = [&a,&c](double x) -> double { return a * (x - c) * (x - c); };
  auto dphi = [&a,&c](double x) -> double { return 2 * a * (x - c); };

  std::cout << "\n#### Single Iter ####\n";
  linesearch::CubicLineSearch ls;
  ls.SetVerbose(true);
  double phi0 = phi(0.0);
  double dphi0 = dphi(0.0);
  double alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 1);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  // Change the center slightly
  //   Without changing c2, it will still only take 1 iteration
  std::cout << "\n#### Single Iter, But Off ####\n";
  c = 1.1;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 1);
  EXPECT_DOUBLE_EQ(alpha, 1.0);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  //  Changing c2 to a tighter tolerance will force it to find the true minimum
  std::cout << "\n#### Use Cubic Interp ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.01);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 3);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());
  std::cout << "alpha = " << alpha << std::endl;

  // Overshoot, grow maximum, and enter condition where alo > ahi
  std::cout << "\n#### Overshoot, ahi < alo ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.1);
  c = 0.8;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_DOUBLE_EQ(alpha, c);
  std::cout << "status = " << ls.StatusToString() << std::endl;
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());


  // Try flipping the quadratic (always decreasing)
  std::cout << "\n#### Hit Max Alpha ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.9);
  a = -1;
  c = -0.1;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_FALSE(ls.CurvatureConditionSatisfied());
  EXPECT_DOUBLE_EQ(alpha, ls.alpha_max);
  std::cout << "status = " << ls.StatusToString() << std::endl;
  std::cout << "alpha = " << alpha << std::endl;
}

TEST(LineSearchTests, Cubic) {
  double c = 1;
  auto phi = [&c](double x) -> double { return (x - c) * (x - c) - (x - c) * (x - c) * (x - c); };
  auto dphi = [&c](double x) -> double { return 2 * (x - c) - 3 * (x - c) * (x - c); };

  std::cout << "\n#### Single Iter ####\n";
  linesearch::CubicLineSearch ls;
  ls.SetVerbose(true);
  double phi0 = phi(0.0);
  double dphi0 = dphi(0.0);
  double alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 1);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  std::cout << "\n#### Tight tolerance to 1.2 ####\n";
  ls.SetOptimalityTolerances(1e-4, 1e-3);
  c = 1.2;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 3);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  std::cout << "\n#### c = 1.8 ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.01);
  c = 1.8;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 4);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  std::cout << "\n#### c = 0.8 ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.01);
  c = 0.8;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 2);
  EXPECT_DOUBLE_EQ(alpha, c);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());

  std::cout << "\n#### c = 0.01 ####\n";
  ls.SetOptimalityTolerances(1e-4, 0.01);
  c = 0.01;
  phi0 = phi(0.0);
  dphi0 = dphi(0.0);
  alpha = ls.Run(phi, dphi, 1.0, phi0, dphi0);
  EXPECT_EQ(ls.Iterations(), 2);
  std::cout << "Iterations = " << ls.Iterations() << std::endl;
  EXPECT_NEAR(alpha, c, 1e-6);
  EXPECT_EQ(ls.GetStatus(), linesearch::CubicLineSearch::ReturnCodes::MINIMUM_FOUND);
  EXPECT_TRUE(ls.SufficientDecreaseSatisfied());
  EXPECT_TRUE(ls.CurvatureConditionSatisfied());
}