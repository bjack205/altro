cmake_minimum_required(VERSION 3.13)
list(APPEND CMAKE_MESSAGE_CONTEXT altro)
project(altro VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)

#############################################
# Options
#############################################

# Handle default build type
set(ALTRO_DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Setting CMAKE_BUILD_TYPE to ${ALTRO_DEFAULT_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE ${ALTRO_DEFAULT_BUILD_TYPE} CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable testing
option(ALTRO_BUILD_TESTS "Build tests for altro" ON)

# Code Coverage
option(ALTRO_CODECOV "Compile altro with code coverage" OFF)
if (ALTRO_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "Compiler altro with coverage info.")
  add_compile_options(-O0 -g -fprofile-arcs -ftest-coverage)
endif()

# Documentation
option(ALTRO_BUILD_DOCS "Build documentation for altro." OFF)

# Use -fPIC option
option(ALTRO_POSITION_INDEPENDENT_CODE "Use -fPIC flag when compiling altro" ON)

#############################################
# Dependencies
#############################################
include(FetchContent)

# Add libfmt
list(APPEND CMAKE_MESSAGE_CONTEXT fmt)
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG a33701196adfad74917046096bf5a2aa0ab0bb50  # version 9.1.0
  )
FetchContent_MakeAvailable(fmt)
export_library(fmt AltroCppTargets)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

# Add Eigen
list(APPEND CMAKE_MESSAGE_CONTEXT eigen)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation." FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # version 3.4.0
  )
FetchContent_MakeAvailable(eigen)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

#############################################
# Documentation
#############################################

if (ALTRO_BUILD_DOCS)
  message(STATUS "Adding altro_docs target to build documentation.")
  add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

#############################################
# Build
#############################################

# Output all binaries

set(CMAKE_POSITION_INDEPENDENT_CODE ${ALTRO_USE_FPIC})
