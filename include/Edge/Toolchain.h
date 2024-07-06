// Toolchain.h
// ~~~~~~~~~~~
// Gathers compiler modules and executes them in order.
#ifndef EDGE_TOOLCHAIN_H
#define EDGE_TOOLCHAIN_H

#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <llvm/IR/Module.h>

#include <filesystem>

static std::error_code EC;

namespace edge {

enum class CompilationStrategy { LLVM, MLIR, NATIVE };

class Toolchain {
 private:
  const char *fileName;
  bool shouldEmit;
  llvm::raw_fd_ostream *EmitFile = nullptr;
  CompilationStrategy strategy;

  Lexer *lexer;
  Parser *parser;

  void initFrontend() {
    lexer = new Lexer(fileName);
    parser = new Parser(lexer);
  }

  std::string getEmitFileSuffix() const {
    switch (strategy) {
      case CompilationStrategy::NATIVE:
        return ".s";
      case CompilationStrategy::LLVM:
        return ".ll";
      default:
        return ".mlir";
    }
  }

  // Specific toolchain executors.
  void executeMLIRToolchain();
  void executeLLVMToolchain();
  void executeNativeToolchain();

 public:
  Toolchain(const char *fileName,
            CompilationStrategy strategy = CompilationStrategy::MLIR,
            bool shouldEmit = false);

  ~Toolchain() {
    delete lexer;
    delete parser;
    if (EmitFile) {
      EmitFile->flush();
      EmitFile->close();
      delete EmitFile;
    }
  }
  // Get edge filename
  const char *getFileName() const { return fileName; }

  static std::unique_ptr<llvm::Module> llvmModuleFromASTQuick(
      ProgramAST *AST, llvm::LLVMContext &ctx) {
    LLVMGenerator generator(ctx);
    return std::move(generator.codeGenModule(*AST));
  }

  void execute();
};

}  // namespace edge
#endif  // EDGE_TOOLCHAIN_H
