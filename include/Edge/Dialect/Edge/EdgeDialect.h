// Dialect.h
// ~~~~~~~~~
// Header file for the EdgeDialect
#ifndef EDGE_DIALECT_EDGE_DIALECT_H
#define EDGE_DIALECT_EDGE_DIALECT_H
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>

#include <Edge/Dialect/Edge/EdgeDialect.h.inc>

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#define GET_OP_CLASSES
#include <Edge/Dialect/Edge/EdgeOps.h.inc>

#define CONSTANT_OP_WIDTH 64

#define GET_SI64_TYPE(BUILDER) BUILDER.getIntegerType(CONSTANT_OP_WIDTH, true)

using EdgeSymbolTable = std::map<llvm::StringRef, mlir::Value>;

#endif  // EDGE_DIALECT_EDGE_DIALECT_H
