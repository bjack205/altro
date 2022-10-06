//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "Eigen/Dense"
#include "altro/utils/formatting.hpp"
#include "altro/solver/typedefs.hpp"

void discrete_double_integrator_dynamics(double *xnext, const double *x, const double *u, float h,
                                         int dim);

void discrete_double_integrator_jacobian(double *jac, const double *x, const double *u, float h,
                                         int dim);

void cartpole_dynamics_midpoint(double *xnext, const double *x, const double *u, float h);
void cartpole_jacobian_midpoint(double *xnext, const double *x, const double *u, float h);

void pendulum_dynamics(double *xnext, const double *x, const double *u);
void pendulum_jacobian(double *jac, const double *x, const double *u);

using ContinuousDynamicsFunction = std::function<void(double*, const double*, const double*)>;
using ContinuousDynamicsJacobian = std::function<void(double*, const double*, const double*)>;

altro::ExplicitDynamicsFunction MidpointDynamics(int n, int m, ContinuousDynamicsFunction f);
altro::ExplicitDynamicsJacobian MidpointJacobian(int n, int m, ContinuousDynamicsFunction f, ContinuousDynamicsJacobian jac);
