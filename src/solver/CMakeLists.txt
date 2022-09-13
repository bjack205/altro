add_library(altro_impl STATIC
  solver.hpp
  solver.cpp
  internal_types..hpp shifted_vector.cpp shifted_vector.hpp)
target_link_libraries(altro_impl
  fmt::fmt
  Eigen3::Eigen
)
target_include_directories(altro_impl
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_subdirectory(test)