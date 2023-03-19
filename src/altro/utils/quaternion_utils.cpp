//
// Created by Zixin Zhang on 03/17/23.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#include "quaternion_utils.hpp"

namespace altro {
Eigen::Matrix3d skew(Eigen::Vector3d vec) {
    Eigen::Matrix3d skew;
    skew << 0, -vec[2], vec[1],
            vec[2], 0, -vec[0],
            -vec[1], vec[0], 0;
    return skew;
}

Eigen::Vector4d cayley_map(Eigen::Vector3d phi) {
    Eigen::Vector4d phi_quat;
    phi_quat << 1, phi[0], phi[1], phi[2];
    return 1 / (sqrt(1 + phi.norm() * phi.norm())) * phi_quat;
}

Eigen::Vector3d inv_cayley_map(Eigen::Vector4d q) {
    return q.tail(3) / q[0];
}

Eigen::Vector4d quat_conj(Eigen::Vector4d q) {
    Eigen::Matrix4d T;
    T << 1, 0, 0, 0,
         0, -1, 0, 0,
         0, 0, -1, 0,
         0, 0, 0, -1;
    return T * q;
}

Eigen::Matrix4d L(Eigen::Vector4d q) {
    Eigen::Matrix4d L;
    L(0, 0) = q[0];
    L.block<1, 3>(0, 1) = -q.tail(3).transpose();
    L.block<3, 1>(1, 0) = q.tail(3);
    L.block<3, 3>(1, 1) = q[0] * Eigen::Matrix3d::Identity() + skew(q.tail(3));
    return L;
}

Eigen::Matrix4d R(Eigen::Vector4d q) {
    Eigen::Matrix4d R;
    R(0, 0) = q[0];
    R.block<1, 3>(0, 1) = -q.tail(3).transpose();
    R.block<3, 1>(1, 0) = q.tail(3);
    R.block<3, 3>(1, 1) = q[0] * Eigen::Matrix3d::Identity() - skew(q.tail(3));
    return R;
}

Eigen::MatrixXd G(Eigen::Vector4d q) {
    Eigen::MatrixXd H(4, 3);
    H << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         0, 0, 1;
    return L(q) * H;
}

Eigen::Vector4d quat_mult(Eigen::Vector4d q1, Eigen::Vector4d q2) {
    return L(q1) * q2;
}
} // namespace altro
