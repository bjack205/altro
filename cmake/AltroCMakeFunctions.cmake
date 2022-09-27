function(altro_unit_test testname)
  if (${ALTRO_BUILD_TESTS})
     add_executable(${testname}_test ${testname}_test.cpp)
      target_link_libraries(${testname}_test
        PRIVATE
        gtest_main
        altro::impl
        fmt::fmt
        Eigen3::Eigen
        altro::test_utils
      )
    gtest_discover_tests(${testname}_test)
    target_include_directories(${testname}_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
  endif()
endfunction()