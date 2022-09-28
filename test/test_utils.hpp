//
// Created by Brian Jackson on 9/27/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "Eigen/Dense"
#include "altro/utils/formatting.hpp"

void discrete_double_integrator_dynamics(double *xnext, const double *x, const double *u, float h,
                                         int dim);

void discrete_double_integrator_jacobian(double *jac, const double *x, const double *, float h,
                                         int dim);
