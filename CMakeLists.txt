cmake_minimum_required(VERSION 3.29)

project(nn++ LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)

add_executable(nn++)

target_sources(
    nn++
    PUBLIC
    main.cpp
)


target_include_directories(nn++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(nn++ PRIVATE -fsanitize=address)
target_link_options(nn++ PRIVATE -fsanitize=address)

find_package(structopt REQUIRED)
target_link_libraries(nn++ PRIVATE structopt::structopt)
