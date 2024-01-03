// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Frontend.h>
#include <Toolchain.h>

namespace edge {

void Toolchain::executeToolchain() {
  ProgramAST *AST = new ProgramAST();
  bool parse = parser->parseProgram(AST);
  std::cout << parse << std::endl;
}

}  // namespace edge
