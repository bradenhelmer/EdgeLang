// main.cpp
// ~~~~~~~~
// Main entrypoint for compiler.

#include <Edge/Toolchain.h>
#include <llvm/Support/CommandLine.h>

#include <iostream>

using namespace edge;
namespace cl = llvm::cl;

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"));
namespace {
enum CompilationStrategy { LLVM, MLIR };

static cl::opt<enum CompilationStrategy> CompilationStrategy(
    "cs", cl::init(MLIR),
    cl::desc("Which compilation strategy should we choose, LLVM or MLIR."),
    cl::values(clEnumValN(LLVM, "llvm",
                          "Compile the program using LLVM only.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "Compile the program using MLIR/LLVM.")));
}  // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "EdgeLang Compiler");

  Toolchain *TC = new Toolchain(InputFilename.getValue().c_str());

  if (CompilationStrategy == MLIR)
    TC->executeMLIRToolchain();
  else
    TC->executeLLVMToolChain();

  delete TC;
  return 0;
}
