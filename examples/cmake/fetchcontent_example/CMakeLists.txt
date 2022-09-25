cmake_minimum_required(VERSION 3.19)
project(AltroFetchContentExample)

include(FetchContent)
list(APPEND CMAKE_MESSAGE_CONTEXT altro)
FetchContent_Declare(altro
  GIT_REPOSITORY https://github.com/bjack205/altro
  GIT_TAG 9819e66a6d20daf39e23bd9b45444b7f5aeb3322
  )
FetchContent_MakeAvailable(altro)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE altro::altro fmt::fmt Eigen3::Eigen)
