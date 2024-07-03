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

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"));
namespace {
enum CompilationStrategy { LLVM, MLIR, NATIVE };

static cl::opt<enum CompilationStrategy> CompilationStrategy(
    "cs", cl::init(MLIR),
    cl::desc("Which compilation strategy should we choose, LLVM or MLIR."),
    cl::values(clEnumValN(LLVM, "llvm",
                          "Compile the program using LLVM only.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "Compile the program using MLIR/LLVM.")),
    cl::values(clEnumValN(NATIVE, "native",
                          "Compile the program using LLVM IR without LLVM backend.")));
}  // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "EdgeLang Compiler");

  Toolchain *TC = new Toolchain(InputFilename.getValue().c_str());

  switch (CompilationStrategy) {
	case NATIVE:
	  TC->executeNativeToolchain();
	  break;
	case LLVM:
	  TC->executeLLVMToolchain();
	  break;
	default:
	  TC->executeMLIRToolchain();
	  break;
  }

  delete TC;
  return 0;
}
