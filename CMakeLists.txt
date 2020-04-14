cmake_minimum_required(VERSION 3.15)
project(matslise_benchmark)

set(CMAKE_CXX_STANDARD 14)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif ()

find_path(MATSLISE_INCLUDE_DIR matslise)
find_library(MATSLISE_LIBRARY matslise)
include_directories(${MATSLISE_INCLUDE_DIR})

find_package(Eigen3 3.3 REQUIRED)

add_compile_definitions(CATCH_CONFIG_ENABLE_BENCHMARKING)
add_executable(matslise_benchmark src/main.cpp src/matslise.cpp)
target_link_libraries(matslise_benchmark Eigen3::Eigen ${MATSLISE_LIBRARY})