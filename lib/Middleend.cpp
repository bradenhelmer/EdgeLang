// Middleend.cpp
// ~~~~~~~~~~~~~
// Handles the generation of Edge MLIR.
#include <Edge/Middleend.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace edge {

mlir::ModuleOp MLIRGenerator::genModuleOp(ProgramAST &ast) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(theModule.getBody());

  /* mlir::Location nullLoc = mlir::FileLineColLoc::get( */
  /*     mlir::StringAttr::get("NULL", builder.getI8Type()), 1, 1); */

  auto dataType = mlir::IntegerType::get(builder.getContext(), 64);
  auto attr = mlir::IntegerAttr::get(dataType, 100);

  builder.create<ConstantOp>(builder.getUnknownLoc(), dataType, attr);
  return theModule;
}

edge::AssignOp MLIRGenerator::genAssignOp(AssignExpr &assignExpr) {
  return nullptr;
}

}  // namespace edge
