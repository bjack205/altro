
# Set directory for documentation to be the one for the current project (not top-level project)
# See Professional CMake 26.1.1
set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PROJECT_NAME})

# Install targets
#install(TARGETS fmt EXPORT AltroCppTargets)

# Add interface to altrocpp export
#set_target_properties(altrocpp_interface PROPERTIES
#  EXPORT_NAME interface
#)
#install(TARGETS altrocpp_interface EXPORT AltroCppTargets)  # legal after 3.14

# Install targets from this project
set(ALTRO_INSTALL_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})  # location of version and config files

install(TARGETS altro_altro2 altro_tvlqr linesearch_linesearch
  EXPORT AltroTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT AltroRuntime
)
if (NOT BUILD_SHARED_LIBS)
  install(TARGETS altro_impl EXPORT AltroTargets COMPONENT AltroRuntime)
endif()
install(EXPORT AltroTargets
  DESTINATION ${ALTRO_INSTALL_CMAKE_DIR}
  NAMESPACE ${PROJECT_NAME}::
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/altroConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake
  INSTALL_DESTINATION ${ALTRO_INSTALL_CMAKE_DIR}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/altroConfigVersion.cmake
  DESTINATION ${ALTRO_INSTALL_CMAKE_DIR}
  COMPONENT AltroRuntime
)


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
