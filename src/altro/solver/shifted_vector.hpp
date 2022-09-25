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

  ShiftedVector(const ShiftedVector& other) = default;
  ShiftedVector(ShiftedVector&& other) = default;
  ShiftedVector& operator=(const ShiftedVector& other) = default;
  ShiftedVector& operator=(ShiftedVector&& other) = default;

  // Create from iterators
  template <class Iterator>
  ShiftedVector(const Iterator start, const Iterator end)
      : len_(end - start), start_index_(0), data_(start, end) {}

  // Create from any type that defines iterators
  template <class Vector>
  ShiftedVector(const Vector& vec)
      : len_(vec.size()), start_index_(0), data_(vec.begin(), vec.end()) {}
  ~ShiftedVector() = default;

  // Index shift
  void SetStart(int i) { start_index_ = ((i + len_) % len_); }
  void ShiftStart(int shift = 1) { start_index_ = (start_index_ + shift + len_) % len_; }
  int size() const { return len_; }

  T& operator[](int i) { return data_[GetTrueIndex(i)]; }

  const T& operator[](int i) const { return data_[GetTrueIndex(i)]; }

  // TODO (brian): add iteration

 private:
  int GetTrueIndex(int i) {
    // Let std::vector handle the bounds checking
    return (start_index_ + i) % len_;
  }

  const int len_;
  int start_index_;
  std::vector<T> data_;
};

}  // namespace altro