cmake_minimum_required(VERSION 3.15)
project(matslise_benchmark)

set(CMAKE_CXX_STANDARD 20)

if (NOT MSVC)
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif ()

set(MATSLISE_PYTHON OFF)
set(MATSLISE_QUADMATH ON)

add_subdirectory(lib/matslise)
include_directories(lib/matslise)

find_package(Eigen3 3.3 REQUIRED)

add_compile_definitions(CATCH_CONFIG_ENABLE_BENCHMARKING)
add_executable(matslise_benchmark src/main.cpp src/matslise.cpp)
target_link_libraries(matslise_benchmark Eigen3::Eigen matslise)

add_executable(matslise_float128 src/main_float128.cpp)
target_link_libraries(matslise_float128 Eigen3::Eigen matslise)

add_executable(paine_float128 src/paine_float128.cpp)
target_link_libraries(paine_float128 Eigen3::Eigen matslise)
