// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <Edge/Toolchain.h>
#include <mlir/IR/MLIRContext.h>

using namespace mlir::edge;

namespace edge {

void Toolchain::executeToolchain() {
  ProgramAST *AST = new ProgramAST();
  bool parse = parser->parseProgram(AST);

  std::printf(
      "This is a test to see if everything is linking properly:\nThe namespace "
      "of the dialect is: %s\n",
      EdgeDialect::getDialectNamespace().str().c_str());

  mlir::MLIRContext context;
  MLIRGenerator generator(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = generator.genModuleOp(*AST);
  module->dump();

  delete AST;
}
}  // namespace edge
