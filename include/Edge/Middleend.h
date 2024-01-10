// Middleend.h
// ~~~~~~~~~~~
// Definitions for the middle end of the compiler:
// 1. Edge Dialect
#ifndef EDGE_MIDDLEEND_H
#define EDGE_MIDDLEEND_H
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>

namespace edge {

class MLIRGenerator {
 private:
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  // Operation Gen
  edge::AssignOp genAssignOp(AssignStmt &assignExpr);
  edge::ConstantOp genConstantOp(IntegerLiteralExpr &integerLitExpr);
  edge::OutputOp genOutputOp(OutputStmt &outputStmt);
  edge::RefOp genRefOp(AssigneeReferenceExpr &refExpr);
  mlir::Value genBinOp(BinaryOpExpr &binOp);
  mlir::Value genExpr(Expr &expr);

 public:
  MLIRGenerator(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp genModuleOp(ProgramAST &ast);
};
}  // namespace edge
#endif  // EDGE_MIDDLEEND_H
