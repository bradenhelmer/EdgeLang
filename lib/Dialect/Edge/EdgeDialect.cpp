// EdgeDialect.cpp
// ~~~~~~~~~~~~~~~
// Implementation of the Edge dialect
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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

static const llvm::StringRef symbolRefToStringRef(SymbolRefAttr attr) {
  return attr.getLeafReference().getValue();
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
void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value) {
  auto dataType = GET_SI64_TYPE(odsBuilder);
  auto attr = IntegerAttr::get(dataType, value);
  ConstantOp::build(odsBuilder, odsState, dataType, attr);
}

LogicalResult ConstantOp::verify() {
  auto type = getResult().getType();
  auto result = llvm::dyn_cast<IntegerType>(type);
  auto width = type.getWidth() == CONSTANT_OP_WIDTH;
  auto signedSemantics =
      type.getSignedness() == IntegerType::SignednessSemantics::Signed;
  if (result && width && signedSemantics) {
    return success();
  } else {
    return emitOpError(
               "ConstantOp could not be verified as a 64 bit signed integer! "
               "It was a ")
           << type.getWidth() << " bit "
           << getSignedSemanticsString(type.getSignedness()) << " integer!";
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValueAttr();
}

// AddOp
// ~~~~~
void AddOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(GET_SI64_TYPE(odsBuilder));
  odsState.addOperands({lhs, rhs});
}

// SubOp
// ~~~~~
void SubOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(GET_SI64_TYPE(odsBuilder));
  odsState.addOperands({lhs, rhs});
}

// MulOp
// ~~~~~
void MulOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(GET_SI64_TYPE(odsBuilder));
  odsState.addOperands({lhs, rhs});
}

// DivOp
// ~~~~~
void DivOp::build(OpBuilder &odsBuilder, OperationState &odsState, Value lhs,
                  Value rhs) {
  odsState.addTypes(GET_SI64_TYPE(odsBuilder));
  odsState.addOperands({lhs, rhs});
}

// RefOp
void RefOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                  llvm::StringRef symbol) {
  SymbolRefAttr symbolRefAttr =
      SymbolRefAttr::get(odsBuilder.getContext(), symbol);
  RefOp::build(odsBuilder, odsState, GET_SI64_TYPE(odsBuilder), symbolRefAttr);
}

#define GET_OP_CLASSES
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>

// Pattern Rewriting: TableGen not working so doing it manually.
LogicalResult DivOp::canonicalize(DivOp op, PatternRewriter &reWriter) {
  Value LHS = op.getOperands()[0];
  Value RHS = op.getOperands()[1];

  RefOp lRef = LHS.getDefiningOp<RefOp>(), rRef = RHS.getDefiningOp<RefOp>();

  if (lRef && rRef) {
    llvm::StringRef lSym = symbolRefToStringRef(lRef.getSymbol()),
                    rSym = symbolRefToStringRef(rRef.getSymbol());

    if (lSym == rSym) {
      reWriter.replaceOp(op,
                         reWriter.create<ConstantOp>(reWriter.getUnknownLoc(),
                                                     static_cast<int64_t>(1)));
      return success();
    }
  }
  ConstantOp lConst = LHS.getDefiningOp<ConstantOp>(),
             rConst = RHS.getDefiningOp<ConstantOp>();
  if (lConst && rConst) {
    int64_t lVal = lConst.getValue(), rVal = rConst.getValue();
    if (lVal == rVal) {
      reWriter.replaceOp(op,
                         reWriter.create<ConstantOp>(reWriter.getUnknownLoc(),
                                                     static_cast<int64_t>(1)));
      return success();
    }
  }
  return failure();
}
