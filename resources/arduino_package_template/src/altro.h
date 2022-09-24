//
// Created by Brian Jackson on 9/24/22.
// Copyright (c) 2022 Robotic Exploration Lab. All rights reserved.
//

#pragma once

#include "ArduinoEigen/ArduinoEigen.h"
#undef ALTRO_USE_MULTITHREADING  // make sure multithreading is disabled
#include "altro/altro.hpp"

int AltroSum(int a, int b);

void SumVectors(double *a, double *b, int len);