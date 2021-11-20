cmake_minimum_required(VERSION 3.00)
project(libnumanalysis)

set(CMAKE_CXX_STANDARD 20)

add_executable(libnumanalysis main.cpp matrix.h equation.h "calculus.h")
