// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
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

  delete AST;
}
}  // namespace edge
