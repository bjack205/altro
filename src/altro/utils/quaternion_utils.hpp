//
// Created by Zixin Zhang on 03/17/23.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "Eigen/Dense"

namespace altro {

Eigen::Matrix3d skew(Eigen::Vector3d vec);
Eigen::Vector4d cayley_map(Eigen::Vector3d phi);
Eigen::Vector3d inv_cayley_map(Eigen::Vector4d q);
Eigen::Vector4d quat_conj(Eigen::Vector4d q);
Eigen::Matrix4d L(Eigen::Vector4d q);
Eigen::Matrix4d R(Eigen::Vector4d q);
Eigen::MatrixXd G(Eigen::Vector4d q);
Eigen::Vector4d quat_mult(Eigen::Vector4d q1, Eigen::Vector4d q2);
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

} // namespace altro
