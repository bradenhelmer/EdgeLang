// Toolchain.h
// ~~~~~~~~~~~
// Gathers compiler modules and executes them in order.
#ifndef EDGE_TOOLCHAIN_H
#define EDGE_TOOLCHAIN_H

#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <llvm/IR/Module.h>

namespace edge {

class Toolchain {
 private:
  const char *fileName;

  Lexer *lexer;
  Parser *parser;

  void initFrontend() {
    lexer = new Lexer(fileName);
    parser = new Parser(lexer);
  }

 public:
  Toolchain(const char *fileName) : fileName(fileName) { initFrontend(); }

  ~Toolchain() {
    delete lexer;
    delete parser;
  }

  void executeMLIRToolchain();
  void executeLLVMToolchain();
  void executeNativeToolchain();

  const char *getFileName() const { return fileName; }
  static std::unique_ptr<llvm::Module> moduleFromASTQuick(
      ProgramAST *AST, llvm::LLVMContext &ctx) {
    LLVMGenerator generator(ctx);
    return std::move(generator.codeGenModule(*AST));
  }
};

}  // namespace edge
#endif  // EDGE_TOOLCHAIN_H
