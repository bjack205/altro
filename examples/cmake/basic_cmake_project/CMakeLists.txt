cmake_minimum_required(VERSION 3.19)
project(AltroCMakeExample)

find_package(altro 0.1 REQUIRED)
find_package(fmt REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE altro::altro fmt::fmt Eigen3::Eigen)