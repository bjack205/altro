add_library(altrocpp_interface
  altrocpp_interface.hpp
  altrocpp_interface.cpp
)
target_link_libraries(altrocpp_interface
  PUBLIC
  altrocpp::altro
  Eigen3::Eigen

  PRIVATE
  fmt::fmt
)
target_include_directories(altrocpp_interface
  PRIVATE
  $<BUILD_INTERFACE:${altro_SOURCE_DIR}/src>
)

add_library(altrocpp::interface ALIAS altrocpp_interface)

add_subdirectory(test)