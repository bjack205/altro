add_library(altro_impl STATIC
  internal_types.hpp

  solver.hpp
  solver.cpp

  shifted_vector.hpp
  shifted_vector.cpp

  knotpoint_data.hpp
  knotpoint_data.cpp
)
target_link_libraries(altro_impl
  fmt::fmt
  Eigen3::Eigen
  altro::altro
  altrocpp::altro
)
target_include_directories(altro_impl
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_subdirectory(test)