include(GNUInstallDirs)
set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME})  # Professional CMake 26.1.1

# Steps:
# 1. Clone ArduinoEigen library
# 2. Copy Source code from ArduinoEigen library to install location  ALTRO_ARDUINO_LIB_PATH/src/Eigen
# 3. Copy ArduinoEigen header files to src/ArduinoEigen
# 4. Copy fmt include/ directory to src/
# 5. Copy altro source code to src/

# Inputs:
# altro root directory
# fmt source dir
# library dir
if (ALTRO_ENABLE_ARDUINO_LIB_INSTALL)
  add_custom_target(install_arduino_lib
    COMMAND ${CMAKE_COMMAND} "-D ALTRO_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
                             "-D FMT_SOURCE_DIR=${fmt_SOURCE_DIR}"
                             "-D EIGEN_SOURCE_DIR=${arduinoeigen_SOURCE_DIR}"
                             "-D LIB_INSTALL_PATH=${ALTRO_ARDUINO_LIB_INSTALL_DIR}"
                             "-P ${CMAKE_CURRENT_SOURCE_DIR}/AltroArduinoInstall.cmake"
  )
endif()
