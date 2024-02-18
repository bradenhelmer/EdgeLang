// Middleend.cpp
// ~~~~~~~~~~~~~
// Handles the generation of Edge MLIR.
#include <Edge/Middleend.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>

namespace edge {

mlir::ModuleOp MLIRGenerator::genModuleOp(ProgramAST &ast) {
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> globalScope(
      symbolTable);

  builder.setInsertionPointToStart(theModule.getBody());

  // Create main function
  mlir::FunctionType fType = builder.getFunctionType({}, {});
  mlir::func::FuncOp mainFunction =
      mlir::func::FuncOp::create(builder.getUnknownLoc(), "main", fType);
  mainFunction.setPublic();
  theModule.push_back(mainFunction);

  mlir::Block *block = mainFunction.addEntryBlock();
  builder.setInsertionPointToStart(block);

  for (AssignStmt *AE : ast.getAssignExprs()) {
    genAssignOp(*AE);
  }

  genOutputOp(ast.getOutputStmt());

  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return theModule;
}

AssignOp MLIRGenerator::genAssignOp(AssignStmt &assignStmt) {
  llvm::StringRef assignee = assignStmt.getAssignee();

  mlir::Value exprGen = genExpr(assignStmt.getExpr());

  if (!exprGen) {
    mlir::emitError(builder.getUnknownLoc(), "Error generating AssignOp!");
    return nullptr;
  }

  symbolTable.insert(assignee, exprGen);
  mlir::StringAttr assigneeStrAttr =
      mlir::StringAttr::get(builder.getContext(), assignee);
  builder.create<AssignOp>(builder.getUnknownLoc(), assigneeStrAttr, exprGen);

  return nullptr;
}

ConstantOp MLIRGenerator::genConstantOp(IntegerLiteralExpr &integerLitExpr) {
  return builder.create<ConstantOp>(builder.getUnknownLoc(),
                                    integerLitExpr.getValue());
}

OutputOp MLIRGenerator::genOutputOp(OutputStmt &outputStmt) {
  mlir::Value outputExprGen = genExpr(outputStmt.getExpr());
  return builder.create<OutputOp>(builder.getUnknownLoc(), outputExprGen);
}

RefOp MLIRGenerator::genRefOp(AssigneeReferenceExpr &refExpr) {
  auto assignee = refExpr.getAssignee();
  if (!symbolTable.lookup(assignee)) {
    mlir::emitError(
        builder.getUnknownLoc(),
        "Reference operation trying to reference a symbol not in scope!");
    return nullptr;
  }
  return builder.create<RefOp>(builder.getUnknownLoc(), refExpr.getAssignee());
}

mlir::Value MLIRGenerator::genBinOp(BinaryOpExpr &binOp) {
  mlir::Value LHS = genExpr(binOp.getLHS());
  mlir::Value RHS = genExpr(binOp.getRHS());
  if (!LHS || !RHS) {
    mlir::emitError(builder.getUnknownLoc(),
                    "Error generating right or left side of binary operation!");
    return nullptr;
  }

  switch (binOp.getOp()) {
    case ADD:
      return builder.create<AddOp>(builder.getUnknownLoc(), LHS, RHS);
    case SUB:
      return builder.create<SubOp>(builder.getUnknownLoc(), LHS, RHS);
    case MUL:
      return builder.create<MulOp>(builder.getUnknownLoc(), LHS, RHS);
    case DIV:
      return builder.create<DivOp>(builder.getUnknownLoc(), LHS, RHS);
    default:
      return nullptr;
  }
}

mlir::Value MLIRGenerator::genExpr(Expr &expr) {
  switch (expr.getType()) {
    case Expr::INTEGER_LITERAL:
      return genConstantOp(dynamic_cast<IntegerLiteralExpr &>(expr));
    case Expr::ASSIGNEE_REF:
      return genRefOp(dynamic_cast<AssigneeReferenceExpr &>(expr));
    case Expr::BINOP:
      return genBinOp(dynamic_cast<BinaryOpExpr &>(expr));
    default:
      return nullptr;
  }
}
}  // namespace edge
