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

  exceptions.hpp
  exceptions.cpp
)
target_link_libraries(altro_impl
  PUBLIC
  Eigen3::Eigen
  altrocpp::altro
  altrocpp::interface

  PRIVATE
  fmt::fmt
  altro::tvlqr
  linesearch::linesearch
)

# Make all includes relative to source directory
target_include_directories(altro_impl
  PUBLIC
  $<BUILD_INTERFACE:${altro_SOURCE_DIR}/src>
)

set_target_properties(altro_impl PROPERTIES EXPORT_NAME impl)
add_library(altro::impl ALIAS altro_impl)

add_subdirectory(test)