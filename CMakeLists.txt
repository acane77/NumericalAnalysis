cmake_minimum_required(VERSION 3.00)
project(libnumanalysis)

set(CMAKE_CXX_STANDARD 20)

if (MSVC) 
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest/googletest/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/gtest)

add_executable(test_equation ${CMAKE_CURRENT_SOURCE_DIR}/example/test_equation.cpp)
target_link_libraries(test_equation gtest_main)

add_executable(test_calculus ${CMAKE_CURRENT_SOURCE_DIR}/example/test_calculus.cpp)
target_link_libraries(test_calculus gtest_main)

add_executable(test_interpolation ${CMAKE_CURRENT_SOURCE_DIR}/example/test_interpolation.cpp)
target_link_libraries(test_interpolation gtest_main)

add_executable(test_matrix ${CMAKE_CURRENT_SOURCE_DIR}/example/test_matrix.cpp)
target_link_libraries(test_matrix gtest_main)
