cmake_minimum_required(VERSION 3.29.0)

project(nn++ LANGUAGES C CXX)

add_compile_options(-Wall -Wextra -Wpedantic)

add_executable(nn++ main.cpp)

target_include_directories(nn++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
