cmake_minimum_required(VERSION 3.16)
project(Basics1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

# JIT - gamma0
add_executable(gamma0 src/gamma0.cpp)
target_link_libraries(gamma0 PRIVATE Halide::Halide Halide::ImageIO)

# AOT - gamma
add_executable(gamma_generator src/HalideGamma.cpp)
target_link_libraries(gamma_generator PRIVATE Halide::Generator)

add_halide_library(gamma FROM gamma_generator FUNCTION_NAME halide_gamma)

add_executable(gamma1 src/gamma.cpp)
target_link_libraries(gamma1 PRIVATE gamma Halide::ImageIO)
