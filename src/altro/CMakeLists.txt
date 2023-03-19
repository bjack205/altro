add_subdirectory(solver)

add_library(altro_altro2

  altro.hpp
  altro.cpp

  altro_solver.hpp
  altro_solver.cpp

  utils/formatting.hpp
  utils/quaternion_utils.hpp
  utils/quaternion_utils.cpp
)
target_link_libraries(altro_altro2
  PRIVATE
  fmt::fmt
  altro_impl
)
target_include_directories(altro_altro2
  PUBLIC
  $<BUILD_INTERFACE:${altro_SOURCE_DIR}/src>
)
set_target_properties(altro_altro2 PROPERTIES EXPORT_NAME altro)
add_library(altro::altro ALIAS altro_altro2)


target_sources(altro_altro2
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ..
  FILES
  altro.hpp
  altro_solver.hpp
  solver/solver_options.hpp
  solver/exceptions.hpp
  solver/typedefs.hpp
)