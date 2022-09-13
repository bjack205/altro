add_library(altro
  typedefs.hpp

  altro.hpp
  altro.cpp

  altro_solver.hpp
  altro_solver.cpp

  solver_options.hpp
  solver_options.cpp
)
target_link_libraries(altro
  PRIVATE
  fmt::fmt
  altro_impl
)
target_include_directories(altro
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

add_library(altro::altro ALIAS altro)
