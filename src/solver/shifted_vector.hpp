//
// Created by Brian Jackson on 9/12/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include <stdexcept>
#include <vector>

namespace altro {

template <class T>
class ShiftedVector {
 public:
  ShiftedVector(int length, const T& value = T())
      : len_(length), start_index_(0), data_(length, value) {}

  // Index shfit
  void SetStart(int i) { start_index_ = ((i + len_) % len_); }
  void ShiftStart(int shift = 1) { start_index_ = (start_index_ + shift + len_) % len_; }
  int size() const { return len_; }

  T& operator[](int i) {
    return data_[GetTrueIndex(i)];
  }

  const T& operator[](int i) const {
    return data_[GetTrueIndex(i)];
  }



 private:
  int GetTrueIndex(int i) {
    if (i < 0) {
      throw(std::range_error("Index must be positive."));
    } else if (i > len_) {
      throw(std::range_error("Index out of range."));
    }
    return (start_index_ + i) % len_;
  }

  const int len_;
  int start_index_;
  std::vector<T> data_;
};

}  // namespace altro