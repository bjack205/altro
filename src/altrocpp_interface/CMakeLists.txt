add_library(altrocpp_interface
  altrocpp_interface.hpp
  altrocpp_interface.cpp
)
target_link_libraries(altrocpp_interface
  PUBLIC
  altrocpp::altro
  altro::altro2
  Eigen3::Eigen

  PRIVATE
  fmt::fmt
)
target_include_directories(altrocpp_interface
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(altrocpp::interface ALIAS altrocpp_interface)

#add_subdirectory(test)