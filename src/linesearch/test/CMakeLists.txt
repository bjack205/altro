add_executable(linesearch_tests
  linesearch_tests.cpp
)
target_link_libraries(linesearch_tests
  PRIVATE
  gtest_main
  linesearch::linesearch
)
