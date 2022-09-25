add_library(altro_impl STATIC
  typedefs.hpp
  internal_types.hpp

  solver.hpp
  solver.cpp

  shifted_vector.hpp
  shifted_vector.cpp

  knotpoint_data.hpp
  knotpoint_data.cpp

  solver_options.hpp
  solver_options.cpp

  solver_stats.cpp
  solver_stats.hpp
)
target_link_libraries(altro_impl
  PUBLIC
  Eigen3::Eigen

  PRIVATE
  fmt::fmt
  altrocpp::altro
  altrocpp::interface
)
target_include_directories(altro_impl
  PUBLIC
  ../..
)

add_subdirectory(test)