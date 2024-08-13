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

  for (AssignStmt *AE : ast.getAssignStmts()) {
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
      return genConstantOp(static_cast<IntegerLiteralExpr &>(expr));
    case Expr::ASSIGNEE_REF:
      return genRefOp(static_cast<AssigneeReferenceExpr &>(expr));
    case Expr::BINOP:
      return genBinOp(static_cast<BinaryOpExpr &>(expr));
    default:
      return nullptr;
  }
}

std::unique_ptr<llvm::Module> LLVMGenerator::codeGenModule(ProgramAST &ast) {
  // Set up main function.
  llvm::FunctionType *MT = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(ctx),
      {llvm::Type::getInt32Ty(ctx),
       llvm::PointerType::get(
           llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0), 0)},
      false);

  llvm::Function *MF = llvm::Function::Create(
      MT, llvm::Function::LinkageTypes::ExternalLinkage, "main", *theModule);
  mainFunction = MF;

  /*MF->args().begin()[0].setName("argc");*/
  /*MF->args().begin()[1].setName("argv");*/

  llvm::BasicBlock *BB = llvm::BasicBlock::Create(ctx, "main_block", MF);
  builder.SetInsertPoint(BB);

  for (const auto &AS : ast.getAssignStmts()) {
    codeGenAssignStmt(*AS);
  }

  codeGenOutputStmt(ast.getOutputStmt());
  builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0));

  return std::move(theModule);
}

void LLVMGenerator::codeGenAssignStmt(const AssignStmt &AS) {
  llvm::AllocaInst *alloca;

  auto assignee = AS.getAssignee();
  llvm::Value *expr = codeGenExpr(AS.getExpr());

  if (!(alloca = static_cast<llvm::AllocaInst *>(
            builder.GetInsertBlock()->getValueSymbolTable()->lookup(
                assignee)))) {
    const auto &prev = builder.GetInsertBlock();
    builder.SetInsertPointPastAllocas(mainFunction);
    alloca = builder.CreateAlloca(llvm::IntegerType::get(ctx, 64), nullptr,
                                  assignee);
    builder.SetInsertPoint(prev);
  }

  builder.CreateStore(expr, alloca);
}

llvm::Value *LLVMGenerator::codeGenExpr(const Expr &E) {
  switch (E.getType()) {
    case Expr::INTEGER_LITERAL:
      return codeGenIntegerLiteral(static_cast<const IntegerLiteralExpr &>(E));
    case Expr::ASSIGNEE_REF:
      return codeGenAssigneeReference(
          static_cast<const AssigneeReferenceExpr &>(E));
    case Expr::BINOP:
      return codeGenBinaryOperation(static_cast<const BinaryOpExpr &>(E));
    default:
      return nullptr;
  }
}

llvm::Value *LLVMGenerator::codeGenIntegerLiteral(
    const IntegerLiteralExpr &IL) {
  return llvm::ConstantInt::get(llvm::IntegerType::get(ctx, 64), IL.getValue());
}

llvm::Value *LLVMGenerator::codeGenAssigneeReference(
    const AssigneeReferenceExpr &AR) {
  llvm::AllocaInst *ref = static_cast<llvm::AllocaInst *>(
      mainFunction->getValueSymbolTable()->lookup(AR.getAssignee()));
  if (!ref) {
    llvm::errs() << "Unknown reference to assignee " << AR.getAssignee()
                 << " \n";
    exit(1);
  }
  return builder.CreateLoad(ref->getAllocatedType(), ref);
}

llvm::Value *LLVMGenerator::codeGenBinaryOperation(const BinaryOpExpr &BO) {
  llvm::Value *LHS = codeGenExpr(BO.getLHS());
  llvm::Value *RHS = codeGenExpr(BO.getRHS());
  if (!LHS || !RHS) {
    llvm::errs()
        << "Error generating left or right side of binary operation!\n";
    exit(1);
  }

  switch (BO.getOp()) {
    case ADD:
      return builder.CreateAdd(LHS, RHS);
    case SUB:
      return builder.CreateSub(LHS, RHS);
    case MUL:
      return builder.CreateMul(LHS, RHS);
    case DIV:
      return builder.CreateSDiv(LHS, RHS);
    default:
      return nullptr;
  }
}

void LLVMGenerator::codeGenOutputStmt(const OutputStmt &OS) {
  // Get printf
  llvm::FunctionType *pf_type = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(ctx), {llvm::PointerType::get(ctx, 0)}, true);

  const auto &fmt = builder.CreateGlobalStringPtr("Result: %d\n", "fmt_spec", 0,
                                                  theModule.get());
  builder.CreateCall(theModule->getOrInsertFunction("printf", pf_type),
                     {fmt, codeGenExpr(OS.getExpr())});
}

void NativeGenerator::lowerLLVMToAssembly() {
  // Get main block
  const auto &mainblock = module->getFunction("main")->getEntryBlock();

  // Construct DAG
  auto DAG = std::make_unique<SelectionDAG>(mainblock);
  DAG->printRaw();
}

}  // namespace edge
