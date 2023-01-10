# add_library(altro_tvlqr
#   tvlqr.h
#   tvlqr.cpp
# )
# target_link_libraries(altro_tvlqr
#   PRIVATE
#   Eigen3::Eigen
#   fmt::fmt
# )
# target_include_directories(altro_tvlqr
#   PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
# )
# add_library(altro::tvlqr ALIAS altro_tvlqr)

# add_subdirectory(test)