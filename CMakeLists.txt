cmake_minimum_required(VERSION 3.22)
project(EdgeLang LANGUAGES CXX C)

# CXX stuff & compile Flags
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_INCLUDE_CURRENT_DIR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -fno-rtti")

# bin aliases
set(EXE ${CMAKE_BINARY_DIR}/edge)
set(DBG lldb)
set(VALGRIND valgrind)

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LLVM_LINK_COMPONENTS Core Support IRReader IRPrinter)

# Source Files
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
file(GLOB SOURCES "${LIB_DIR}/*.cpp")
file(GLOB_RECURSE HEADERS "include/Edge/*.h")

# LLVM / MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Load TableGen and LLVM modules.
include(AddLLVM)
include(TableGen)
# Add MLIR dir path to cmake module path.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_library(edgelib OBJECT ${SOURCES})
set(EDGE_LIBS
    EdgeIR
    EdgePasses
    edgelib
    MLIRExecutionEngine
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs})

# Compiler binary
add_executable(edge $<TARGET_OBJECTS:edgelib>)
target_link_libraries(edge ${EDGE_LIBS})
target_compile_definitions(edge PUBLIC HANDLE_OUTPUT)
# Must compile and TableGen files before building edgelib
add_dependencies(edgelib EdgePasses EdgeIR)

# Targets
add_custom_target(
  run
  COMMAND ${EXE} ${CMAKE_SOURCE_DIR}/test_file.edge --cs llvm
  DEPENDS edge)
add_custom_target(format COMMAND clang-format -i ${SOURCES} ${HEADERS})
add_custom_target(
  dbg
  COMMAND ${DBG} -- ${EXE} --cs llvm ${CMAKE_SOURCE_DIR}/test_file.edge
  DEPENDS edge)
add_custom_target(
  leak_check
  COMMAND ${VALGRIND} --leak-check=full --track-origins=yes ${EXE} --cs llvm
          ${CMAKE_SOURCE_DIR}/test_file.edge
  DEPENDS edge)

add_subdirectory(include)
add_subdirectory(lib)
