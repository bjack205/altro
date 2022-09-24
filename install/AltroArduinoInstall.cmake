cmake_minimum_required(VERSION 3.17)

if (NOT ALTRO_SOURCE_DIR)
  message(FATAL_ERROR "ALTRO_SOURCE_DIR must be set")
endif()

if (NOT FMT_SOURCE_DIR)
  message(FATAL_ERROR "FMT_SOURCE_DIR must be set")
endif()

if (NOT EIGEN_SOURCE_DIR)
  message(FATAL_ERROR "EIGEN_SOURCE_DIR must be set")
endif()

if (NOT LIB_INSTALL_PATH)
  message(FATAL_ERROR "LIB_INSTALL_PATH must be set")
endif()

message(STATUS "Installing arduino library")

# Step 1: Make the directory
set(LIB_NAME "altro")
set(LIB_ROOT ${LIB_INSTALL_PATH}/${LIB_NAME})
file(MAKE_DIRECTORY ${LIB_ROOT})
file(MAKE_DIRECTORY ${LIB_ROOT}/src)

# Step 2: Copy (modified) ArduinoEigen files
file(COPY ${ALTRO_SOURCE_DIR}/resources/arduino_package_template/ DESTINATION ${LIB_ROOT})

# Step 3: Copy ArduinoEigen source files
file(COPY ${EIGEN_SOURCE_DIR}/ArduinoEigen/Eigen DESTINATION ${LIB_ROOT}/src)
file(COPY ${EIGEN_SOURCE_DIR}/ArduinoEigen/utils DESTINATION ${LIB_ROOT}/src/ArduinoEigen)

# Step 4: Copy fmt header files
file(COPY ${FMT_SOURCE_DIR}/include/ DESTINATION ${LIB_ROOT}/src)
