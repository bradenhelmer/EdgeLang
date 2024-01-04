# CMake file for dialect
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(LLVM_TARGET_DEFINITIONS EdgeDialect.td)
mlir_tablegen(EdgeDialect.h.inc -gen-dialect-decls -dialect=edge)
mlir_tablegen(EdgeOps.h.inc -gen-op-decls)
mlir_tablegen(EdgeOps.cpp.inc -gen-op-defs)
mlir_tablegen(EdgeDialect.h.inc -gen-dialect-decls -dialect=edge)
mlir_tablegen(EdgeDialect.cpp.inc -gen-dialect-defs -dialect=edge)

add_mlir_doc(EdgeDialect EdgeDialect Edge/ -gen-dialect-doc)

add_public_tablegen_target(EdgeTableGen)
