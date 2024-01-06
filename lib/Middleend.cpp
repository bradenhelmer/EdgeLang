// Middleend.cpp
// ~~~~~~~~~~~~~
// Handles the generation of Edge MLIR.
#include <Edge/Middleend.h>

namespace edge {

mlir::ModuleOp MLIRGenerator::genModuleOp(ProgramAST &ast) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (AssignExpr *AE : ast.getAssignExprs()) {
    genAssignOp(*AE);
  }
  return theModule;
}

mlir::edge::AssignOp MLIRGenerator::genAssignOp(AssignExpr &assignExpr) {
  return nullptr;
}

}  // namespace edge
