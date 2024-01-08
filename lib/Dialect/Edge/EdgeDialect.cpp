// EdgeDialect.cpp
// ~~~~~~~~~~~~~~~
// Implementation of the Edge dialect
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <Edge/Dialect/Edge/EdgeDialect.cpp.inc>

using namespace mlir;
using namespace edge;

void EdgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>
      >();
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState, int64_t value) {
  auto dataType = IntegerType::get(odsBuilder.getContext(), 64);
  auto attr = IntegerAttr::get(dataType, value);
  ConstantOp::build(odsBuilder, odsState, dataType, attr);
}

#define GET_OP_CLASSES
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>
