cmake_minimum_required(VERSION 3.19)
project(AltroFetchContentExample)
cmake_policy(SET CMP0135 NEW)


include(FetchContent)
list(APPEND CMAKE_MESSAGE_CONTEXT altro)
FetchContent_Declare(altro
  GIT_REPOSITORY https://github.com/bjack205/altro
  GIT_TAG db1a979d29c83bb7d1ef196c4d8b5eac205f5b37  # set this to most recent version
  DOWNLOAD_EXTRACT_TIMESTAMP OFF
)
FetchContent_MakeAvailable(altro)
list(POP_BACK CMAKE_MESSAGE_CONTEXT)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE altro::altro fmt::fmt Eigen3::Eigen)
