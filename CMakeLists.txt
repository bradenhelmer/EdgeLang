cmake_minimum_required(VERSION 3.22)
project(EdgeLang LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)

# CXX stuff
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Compile Flags
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -g -stdlib=libstdc++ -fsantize"
    CACHE STRING "C++ Compiler Flags")

# bin aliases
set(EXE ${CMAKE_BINARY_DIR}/edge)
set(DBG gdb)

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Load TableGen and LLVM modules.
include(AddLLVM)
include(TableGen)

# Source Files
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
file(GLOB SOURCES "${LIB_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "include/Edge/*.h")

add_library(edgelib OBJECT ${SOURCES})
# Add MLIR dir path to cmake module path.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}/include)
target_include_directories(edgelib PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Compiler binary
add_executable(edge $<TARGET_OBJECTS:edgelib>)

# Targets
add_custom_target(run COMMAND ${EXE} ${CMAKE_SOURCE_DIR}/test_file.edge)
add_custom_target(format COMMAND clang-format -i ${SOURCES} ${HEADERS})
add_custom_target(dbg COMMAND ${DBG} -q --args ${EXE}
                              ${CMAKE_SOURCE_DIR}/test_file.edge)

add_subdirectory(include)
add_subdirectory(lib)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON "${CMAKE_SOURCE_DIR}/compile_comands.json")
