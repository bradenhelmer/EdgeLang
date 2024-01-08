// Middleend.h
// ~~~~~~~~~~~
// Definitions for the middle end of the compiler:
// 1. Edge Dialect
#ifndef EDGE_MIDDLEEND_H
#define EDGE_MIDDLEEND_H
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <mlir/IR/Builders.h>

namespace edge {

class MLIRGenerator {
 private:
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;

  edge::AssignOp genAssignOp(AssignExpr &assignExpr);

 public:
  MLIRGenerator(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp genModuleOp(ProgramAST &ast);
};
}  // namespace edge
#endif  // EDGE_MIDDLEEND_H
