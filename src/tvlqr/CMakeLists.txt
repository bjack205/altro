add_library(tvlqr
  tvlqr.h
  tvlqr.cpp
)
target_link_libraries(tvlqr
  PRIVATE
  Eigen3::Eigen
  fmt::fmt
)
target_include_directories(tvlqr
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

add_subdirectory(test)