// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <Edge/Toolchain.h>
#include <mlir/IR/MLIRContext.h>

namespace edge {

void Toolchain::executeToolchain() {
  ProgramAST *AST = new ProgramAST();
  bool parse = parser->parseProgram(AST);

  mlir::MLIRContext context;
  context.loadDialect<EdgeDialect>();

  MLIRGenerator generator(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = generator.genModuleOp(*AST);
  module->dump();

  delete AST;
}
}  // namespace edge
