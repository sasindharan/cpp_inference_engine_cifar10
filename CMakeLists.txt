cmake_minimum_required(VERSION 3.10)
project(CPP_Inference_Engine)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    Operators/include
    utilities/include
    include
)

file(GLOB SOURCES
    src/*.cpp
    Operators/src/*.cpp
    utilities/src/*.cpp
)

add_executable(engine ${SOURCES})