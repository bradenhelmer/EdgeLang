cmake_minimum_required(VERSION 3.22)
project(EdgeLang)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -g -stdlib=libstdc++ -fsantize"
    CACHE STRING "C++ Compiler Flags")

set(EXE ${CMAKE_BINARY_DIR}/edge)
set(DBG gdb)

include_directories(include)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
file(GLOB SOURCES "${SRC_DIR}/*.cpp")
file(GLOB HEADERS "include/*.h")

add_library(edgelib OBJECT ${SOURCES})
target_include_directories(edgelib PRIVATE ${HEADERS})

add_executable(edge $<TARGET_OBJECTS:edgelib>)

add_custom_target(run COMMAND ${EXE} ${CMAKE_SOURCE_DIR}/test_file.edge)
add_custom_target(format COMMAND clang-format -i ${SOURCES} ${HEADERS})
add_custom_target(dbg COMMAND ${DBG} -q --args ${EXE}
                              ${CMAKE_SOURCE_DIR}/test_file.edge)
