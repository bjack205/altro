# Used 3.23 features
# - PROJECT_IS_TOP_LEVEL (3.21)
# - FILE_SETS (3.23)
cmake_minimum_required(VERSION 3.23)
list(APPEND CMAKE_MESSAGE_CONTEXT altro)
project(altro VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

# Organization info
set(ORGANIZATION_NAME_FULL "Robotic Exploration Lab")
set(ORGANIZATION_NAME "rexlab")

# Include needed modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(CMakePrintHelpers)
include(FetchContent)
include(AltroCMakeFunctions)
include(FindThreads)
include(GNUInstallDirs)

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
option(ALTRO_BUILD_TESTS "Build tests for altro" OFF)
if (${ALTRO_BUILD_TESTS})
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG 58d77fa8070e8cec2dc1ed015d66b454c8d78850
    )
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)
  include(CTest)
endif()

# Code Coverage
option(ALTRO_CODECOV "Compile altro with code coverage" OFF)
if (ALTRO_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(STATUS "Compiler altro with coverage info.")
  add_compile_options(-O0 -g -fprofile-arcs -ftest-coverage)
endif()

# Documentation
option(ALTRO_BUILD_DOCS "Build documentation for altro." OFF)

# Use -fPIC option
#option(ALTRO_POSITION_INDEPENDENT_CODE "Use -fPIC flag when compiling altro" ON)
if (BUILD_SHARED_LIBS)
#  set(CMAKE_POSITION_INDEPENDENT_CODE ON FORCE)
  message(STATUS "Building shared library.")
  add_compile_options(-fPIC)
else()
  message(STATUS "Building static library. Note that a shared library is recommended.")
endif()

# Set default install prefix to /opt/rexlab/altro
if(NOT WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND PROJECT_IS_TOP_LEVEL)
  message(STATUS "Setting Altro install prefix to /opt/${ORGANIZATION_NAME}/${PROJECT_NAME}")
  set(CMAKE_INSTALL_PREFIX "/opt/${ORGANIZATION_NAME}/${PROJECT_NAME}" CACHE FILEPATH "Base install location" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OFF)
endif()

# Arduino library installation
option(ALTRO_ENABLE_RUNTIME_EXCEPTIONS "Enable C++ runtime exceptions. If disabled, error return codes will be returned instead." OFF)
option(ALTRO_ENABLE_ARDUINO_LIB_INSTALL "Add the install_arduino_lib target"  OFF)
set(ALTRO_ARDUINO_LIB_INSTALL_DIR $ENV{HOME}/Arduino/libraries CACHE FILEPATH "Directory where the altro Arduino library should be installed")
cmake_print_variables(ALTRO_ARDUINO_LIB_INSTALL_DIR)

#############################################
# Dependencies
#############################################
add_subdirectory(deps)

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
add_compile_options(-Wall -Wextra -pedantic -Werror -march=native)
add_subdirectory(src)

#############################################
# Testing
#############################################

if (${ALTRO_BUILD_TESTS})
  message(STATUS "Building altro testing suite.")
  add_subdirectory(test)
endif()


#############################################
# Installation
#############################################
add_subdirectory(install)