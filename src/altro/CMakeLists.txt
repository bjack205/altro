add_subdirectory(solver)

add_library(altro2

  altro.hpp
  altro.cpp

  altro_solver.hpp
  altro_solver.cpp

  utils/formatting.hpp
)
target_link_libraries(altro2
  PRIVATE
  fmt::fmt
  altro_impl
)
target_include_directories(altro2
  PUBLIC
  $<BUILD_INTERFACE:${altro_SOURCE_DIR}/src>
)

add_library(altro::altro2 ALIAS altro2)
