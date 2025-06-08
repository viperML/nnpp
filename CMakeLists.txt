cmake_minimum_required(VERSION 3.31)

project(nn++ LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)


add_executable(nn++)

target_sources(
    nn++
    PUBLIC
    main.cpp
    matrix.cpp
    mnist.cpp
)

target_compile_options(nn++ PRIVATE -fsanitize=address -g)
target_link_options(nn++ PRIVATE -fsanitize=address -g -lbfd -ldl)
target_include_directories(nn++ PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(nn++ PRIVATE -Wall -pedantic)

find_package(Backward REQUIRED)
target_link_libraries(nn++ PRIVATE Backward::Backward)

enable_testing()
find_package(GTest REQUIRED)
add_executable(matrix_test test/matrix.cpp matrix.cpp)
target_include_directories(matrix_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(matrix_test PRIVATE GTest::gtest_main)
gtest_discover_tests(matrix_test)