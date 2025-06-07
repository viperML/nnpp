cmake_minimum_required(VERSION 3.31)

project(nn++ LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)


add_executable(nn++)

target_sources(
    nn++
    PUBLIC
    main.cpp
    matrix.cpp
)

target_compile_options(nn++ PRIVATE -fsanitize=address)
target_link_options(nn++ PRIVATE -fsanitize=address)
target_include_directories(nn++ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(nn++ PRIVATE -Wall -pedantic)