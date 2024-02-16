// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Edge/Conversion/Edge/Passes.h>
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <Edge/Toolchain.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace edge {

void Toolchain::executeToolchain() {
  ProgramAST *AST = new ProgramAST();
  bool parse = parser->parseProgram(AST);

  mlir::MLIRContext context;
  context.loadDialect<EdgeDialect, mlir::func::FuncDialect>();

  MLIRGenerator generator(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = generator.genModuleOp(*AST);

  mlir::PassManager pm = mlir::PassManager::on<mlir::ModuleOp>(&context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(edge::createIntermediateEdgeLoweringPass());
  pm.addPass(edge::createLLVMIntermediateLoweringPass());
  auto result = pm.run(module.get());

  if (failed(mlir::verify(module.get()))) {
    module.get().emitError("Module verification error!");
  }
  module->dump();

  delete AST;
}
}  // namespace edge
