//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <vector>

#include "Eigen/Dense"
#include "altro/solver/typedefs.hpp"
#include "altro/utils/formatting.hpp"

void discrete_double_integrator_dynamics(double *xnext, const double *x, const double *u, float h,
                                         int dim);

void discrete_double_integrator_jacobian(double *jac, const double *x, const double *u, float h,
                                         int dim);

void cartpole_dynamics_midpoint(double *xnext, const double *x, const double *u, float h);
void cartpole_jacobian_midpoint(double *xnext, const double *x, const double *u, float h);

void pendulum_dynamics(double *xnext, const double *x, const double *u);
void pendulum_jacobian(double *jac, const double *x, const double *u);

using ContinuousDynamicsFunction = std::function<void(double *, const double *, const double *)>;
using ContinuousDynamicsJacobian = std::function<void(double *, const double *, const double *)>;

altro::ExplicitDynamicsFunction MidpointDynamics(int n, int m, ContinuousDynamicsFunction f);
altro::ExplicitDynamicsJacobian MidpointJacobian(int n, int m, ContinuousDynamicsFunction f,
                                                 ContinuousDynamicsJacobian jac);

class BicycleModel {
 public:
  enum class ReferenceFrame { CenterOfGravity, Rear, Front };

  explicit BicycleModel(ReferenceFrame frame = ReferenceFrame::CenterOfGravity)
      : reference_frame_(frame) {}

  void Dynamics(double *x_dot, const double *x, const double *u) const;
  void Jacobian(double *jac, const double *x, const double *u) const;

  void SetLengths(double length, double length_to_rear_wheel_from_cg) {
    length_ = length;
    distance_to_rear_wheels_ = length_to_rear_wheel_from_cg;
  }

  static constexpr int NumStates = 4;
  static constexpr int NumInputs = 2;

 private:
  ReferenceFrame reference_frame_;
  double length_ = 2.7;
  double distance_to_rear_wheels_ = 1.5;
};

void ReadScottyTrajectory(int *Nref, float *tref, std::vector<Eigen::Vector4d> *xref,
                          std::vector<Eigen::Vector2d> *uref);