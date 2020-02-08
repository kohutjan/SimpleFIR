cmake_minimum_required(VERSION 2.8.7)

project(SimpleFIR)

find_package( OpenCV REQUIRED )

set(SimpleFIR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(SimpleFIR_SRC_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories(${SimpleFIR_INCLUDE_DIR})
include_directories(${SimpleFIR_SRC_DIR})

add_executable(simple_fir
	       ${SimpleFIR_SRC_DIR}/main.cpp)

target_link_libraries(simple_fir ${OpenCV_LIBS})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp -std=gnu++17 -Wall -Wextra -pedantic")
