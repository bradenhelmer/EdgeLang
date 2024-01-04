add_mlir_dialect_library(
  EdgeIR
  EdgeDialect.cpp
  DEPENDS
  EdgeTableGen
  LINK_LIBS
  PUBLIC
  MLIRIR)
