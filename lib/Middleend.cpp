// Middleend.cpp
// ~~~~~~~~~~~~~
// Handles the generation of Edge MLIR.
#include <Edge/Middleend.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace edge {

mlir::ModuleOp MLIRGenerator::genModuleOp(ProgramAST &ast) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  mlir::Location nullLoc = mlir::FileLineColLoc::get(
      mlir::StringAttr::get("NULL", builder.getI8Type()), 1, 1);

  /* builder.create<mlir::edge::ConstantOp>(nullLoc, 64); */
  return theModule;
}

edge::AssignOp MLIRGenerator::genAssignOp(AssignExpr &assignExpr) {
  return nullptr;
}

}  // namespace edge
