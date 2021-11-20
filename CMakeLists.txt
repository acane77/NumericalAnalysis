cmake_minimum_required(VERSION 3.00)
project(libnumanalysis)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/example DIR_SRCS)

add_executable(test_equation ${CMAKE_CURRENT_SOURCE_DIR}/example/test_equation.cpp)
target_link_libraries(test_equation)

add_executable(test_calculus ${CMAKE_CURRENT_SOURCE_DIR}/example/test_calculus.cpp)
target_link_libraries(test_calculus)

add_executable(test_interpolation ${CMAKE_CURRENT_SOURCE_DIR}/example/test_interpolation.cpp)
target_link_libraries(test_interpolation)
