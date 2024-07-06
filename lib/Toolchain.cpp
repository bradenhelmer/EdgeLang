// Toolchain.cpp
// ~~~~~~~~~~~~~
// Toolchain implementation
#include <Edge/Common.h>
#include <Edge/Conversion/Edge/Passes.h>
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <Edge/Frontend.h>
#include <Edge/Middleend.h>
#include <Edge/Toolchain.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Transforms/Passes.h>

static llvm::ExitOnError exiter;

namespace cl = llvm::cl;

namespace edge {

Toolchain::Toolchain(const char *fileName, CompilationStrategy strategy,
                     bool shouldEmit)
    : strategy(strategy), fileName(fileName), shouldEmit(shouldEmit) {
  initFrontend();
  if (shouldEmit) {
    auto FileStr = std::string(fileName);
    auto raw = FileStr.substr(0, FileStr.find_last_of('.'))
                   .substr(FileStr.find_first_of('/') + 1);
    EmitFile =
        new llvm::raw_fd_ostream(std::filesystem::current_path().string() +
                                     "/" + raw + getEmitFileSuffix(),
                                 EC);
  }
}

void Toolchain::execute() {
  switch (strategy) {
    case CompilationStrategy::NATIVE:
      executeNativeToolchain();
      break;
    case CompilationStrategy::LLVM:
      executeLLVMToolchain();
      break;
    default:
      executeMLIRToolchain();
      break;
  }
}

void Toolchain::executeNativeToolchain() {
  ProgramAST *AST = new ProgramAST();
  if (!parser->parseProgram(AST)) {
    llvm::errs() << "Error parsing AST!\n";
    exit(1);
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = llvmModuleFromASTQuick(AST, *context);
  module->setSourceFileName(fileName);

  NativeGenerator generator(std::move(module));
  generator.lowerLLVMToAssembly();

  delete AST;
}

void Toolchain::executeLLVMToolchain() {
  ProgramAST *AST = new ProgramAST();
  if (!parser->parseProgram(AST)) {
    llvm::errs() << "Error parsing AST!\n";
    exit(1);
  }

  auto context = std::make_unique<llvm::LLVMContext>();
  auto module = llvmModuleFromASTQuick(AST, *context);
  module->setSourceFileName(fileName);

  if (shouldEmit) {
    module->print(*EmitFile, nullptr);
  }

  if (llvm::verifyModule(*module, &llvm::outs())) {
    llvm::errs() << "Error with IR generation!\n";
    delete AST;
    exit(1);
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::orc::ThreadSafeModule TSM(std::move(module), std::move(context));
  auto JIT = exiter(llvm::orc::LLJITBuilder().create());
  exiter(JIT->addIRModule(std::move(TSM)));
  auto mainf = exiter(JIT->lookup("main"));
  int (*maint)(int, char **) = mainf.toPtr<int(int, char **)>();
  int result = maint(0, nullptr);

  delete AST;
}

void Toolchain::executeMLIRToolchain() {
  ProgramAST *AST = new ProgramAST();
  bool parse = parser->parseProgram(AST);

  mlir::MLIRContext context;
  context.loadDialect<EdgeDialect, mlir::func::FuncDialect>();

  MLIRGenerator generator(context);
  mlir::OwningOpRef<mlir::ModuleOp> module = generator.genModuleOp(*AST);

  if (shouldEmit) {
    *EmitFile << "// Module Phase 1: Edge/Func Dialects\n";
    module->print(*EmitFile);
    *EmitFile << "\n";
    EmitFile->flush();
  }

  mlir::PassManager pm = mlir::PassManager::on<mlir::ModuleOp>(&context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(edge::createIntermediateEdgeLoweringPass());
  if (failed(pm.run(module.get()))) {
    module.get().emitError("Pass error!");
  }
  if (shouldEmit) {
    *EmitFile << "// Module Phase 2: Arith/Func/Memref Dialects\n";
    module->print(*EmitFile);
    *EmitFile << "\n";
    EmitFile->flush();
  }

  pm.clear();
  pm.addPass(edge::createLLVMIntermediateLoweringPass());
  if (failed(pm.run(module.get()))) {
    module.get().emitError("Pass error!");
  }
  if (shouldEmit) {
    *EmitFile << "// Module Phase 3: LLVM Dialect\n";
    module->print(*EmitFile);
  }

  if (failed(mlir::verify(module.get()))) {
    module.get().emitError("Module verification error!");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  auto maybeEngine =
      mlir::ExecutionEngine::create(*module, {.transformer = optPipeline});

  assert(maybeEngine && "Failed to create execution engine!");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invokePacked("main");

  if (invocationResult) {
    llvm::errs() << "JIT failed for error " << invocationResult << "!\n";
  }

  delete AST;
}
}  // namespace edge
