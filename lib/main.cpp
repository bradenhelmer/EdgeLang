// main.cpp
// ~~~~~~~~
// Main entrypoint for compiler.

#include <Edge/Toolchain.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>

#include <iostream>
#include <type_traits>

using namespace edge;
namespace cl = llvm::cl;

namespace edge {
static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"));
static cl::opt<bool> Emit(
    "emit", cl::init(false),
    cl::desc("Optionally MLIR, LLVM IR, or X86-64 Assembly."));

static cl::opt<enum CompilationStrategy> CompilationStrategy(
    "cs", cl::init(CompilationStrategy::MLIR),
    cl::desc("Which compilation strategy should we choose, LLVM or MLIR."),
    cl::values(clEnumValN(CompilationStrategy::LLVM, "llvm",
                          "Compile the program using LLVM only.")),
    cl::values(clEnumValN(CompilationStrategy::MLIR, "mlir",
                          "Compile the program using MLIR/LLVM.")),
    cl::values(
        clEnumValN(CompilationStrategy::NATIVE, "native",
                   "Compile the program using LLVM IR without LLVM backend.")));
}  // namespace edge

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "EdgeLang Compiler");

  Toolchain *TC = new Toolchain(InputFilename.getValue().c_str(),
                                CompilationStrategy, Emit);

  TC->execute();

  delete TC;
  return 0;
}
