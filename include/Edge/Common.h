// Common.h
// ~~~~~~~~
// Common macros and helper functions
#ifndef EDGELANG_COMMON_H
#define EDGELANG_COMMON_H
#include <mlir/Dialect/Arith/IR/Arith.h>

#define MAP_FIND(MAP, KEY) MAP.find(KEY) != MAP.end()

static mlir::arith::ConstantIndexOp zerothIdx = nullptr;

static mlir::arith::ConstantIndexOp &getZeroth(mlir::OpBuilder &builder);

#endif  // EDGELANG_COMMON_H
