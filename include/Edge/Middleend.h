// Middleend.h
// ~~~~~~~~~~~
// Definitions for the middle end of the compiler:
// 1. Edge Dialect
#ifndef EDGE_MIDDLEEND_H
#define EDGE_MIDDLEEND_H
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
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
  MLIRGenerator(mlir::MLIRContext &context) : builder(&context) {
    std::puts("Initializing MLIR Generator...");
  }

  mlir::ModuleOp genModuleOp(ProgramAST &ast);
};

class LLVMGenerator {
 private:
  llvm::IRBuilder<> builder;
  std::unique_ptr<llvm::Module> theModule;
  llvm::LLVMContext &ctx;
  llvm::Function *mainFunction;

  // IR Gen
  void codeGenAssignStmt(const AssignStmt &AS);
  llvm::Value *codeGenExpr(const Expr &E);
  llvm::Value *codeGenIntegerLiteral(const IntegerLiteralExpr &IL);
  llvm::Value *codeGenAssigneeReference(const AssigneeReferenceExpr &AR);
  llvm::Value *codeGenBinaryOperation(const BinaryOpExpr &BO);
  void codeGenOutputStmt(const OutputStmt &OS);

 public:
  LLVMGenerator(llvm::LLVMContext &context)
      : builder(context),
        theModule(std::make_unique<llvm::Module>("EdgeModule", context)),
        ctx(context) {
    std::puts("Initializing LLVM Generator...");
  }
  std::unique_ptr<llvm::Module> codeGenModule(ProgramAST &ast);
};

}  // namespace edge
#endif  // EDGE_MIDDLEEND_H
