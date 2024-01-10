// EdgeDialect.cpp
// ~~~~~~~~~~~~~~~
// Implementation of the Edge dialect
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

#include <Edge/Dialect/Edge/EdgeDialect.cpp.inc>

using namespace mlir;
using namespace edge;

namespace {
static constexpr llvm::StringRef getSignedSemanticsString(
    IntegerType::SignednessSemantics status) {
  switch (status) {
    case IntegerType::Signed:
      return "signed";
    case IntegerType::Unsigned:
      return "unsigned";
    case IntegerType::Signless:
      return "signless";
  }
}
}  // namespace

void EdgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>
      >();
}

// ConstantOp
// ----------
void ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                       ::mlir::OperationState &odsState, int64_t value) {
  auto dataType = IntegerType::get(odsBuilder.getContext(), CONSTANT_OP_WIDTH);
  auto attr = IntegerAttr::get(dataType, value);
  ConstantOp::build(odsBuilder, odsState, dataType, attr);
}

LogicalResult ConstantOp::verify() {
  auto type = getResult().getType();
  auto result = llvm::dyn_cast<IntegerType>(type);
  auto width = type.getWidth() == CONSTANT_OP_WIDTH;
  auto signedSemantics =
      type.getSignedness() == IntegerType::SignednessSemantics::Signed;
  if (!result && width && signedSemantics) {
    return success();
  } else {
    return emitOpError(
               "ConstantOp could not be verified as a 64 bit signed integer! "
               "It was a ")
           << type.getWidth() << " bit "
           << getSignedSemanticsString(type.getSignedness()) << " integer!";
  }
}

// AddOp
// ~~~~~
void AddOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, mlir::Value lhs,
                  mlir::Value rhs) {
  TypeRange valueTypes = {lhs.getType(), rhs.getType()};
  odsState.addTypes(valueTypes);
  odsState.addOperands({lhs, rhs});
}

// SubOp
// ~~~~~
void SubOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, mlir::Value lhs,
                  mlir::Value rhs) {
  TypeRange valueTypes = {lhs.getType(), rhs.getType()};
  odsState.addTypes(valueTypes);
  odsState.addOperands({lhs, rhs});
}

// MulOp
// ~~~~~
void MulOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, mlir::Value lhs,
                  mlir::Value rhs) {
  TypeRange valueTypes = {lhs.getType(), rhs.getType()};
  odsState.addTypes(valueTypes);
  odsState.addOperands({lhs, rhs});
}

// DivOp
// ~~~~~
void DivOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, mlir::Value lhs,
                  mlir::Value rhs) {
  TypeRange valueTypes = {lhs.getType(), rhs.getType()};
  odsState.addTypes(valueTypes);
  odsState.addOperands({lhs, rhs});
}

// RefOp
void RefOp::build(::mlir::OpBuilder &odsBuilder,
                  ::mlir::OperationState &odsState, llvm::StringRef symbol) {
  SymbolRefAttr symbolRefAttr =
      SymbolRefAttr::get(odsBuilder.getContext(), symbol);
  RefOp::build(odsBuilder, odsState, odsBuilder.getI64Type(), symbolRefAttr);
}

#define GET_OP_CLASSES
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>
