//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "solver.hpp"

#include "altrocpp_interface/altrocpp_interface.hpp"
#include "altro/common/knotpoint.hpp"

namespace altro {

bool SolverImpl::Initialize() {
  alsolver_.InitializeFromProblem(problem_);
  is_initialized_ = true;

  // Initialize the trajectory
  using KnotPointXXd = KnotPoint<Eigen::Dynamic, Eigen::Dynamic>;
  std::vector<KnotPointXXd> knotpoints;
  float t = 0.0;
  for (int k = 0; k < horizon_length_ + 1; ++k) {
    VectorXd x(nx_[k]);
    VectorXd u(nu_[k]);
    x.setZero();
    u.setZero();
    KnotPointXXd z(x, u, t);
    knotpoints.push_back(std::move(z));
    t += h_[k];  // note this results in roundoff error
  }

  return is_initialized_;
}

}  // namespace altro
