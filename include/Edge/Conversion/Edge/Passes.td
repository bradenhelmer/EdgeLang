// Passes.td
// ~~~~~~~~~
// Edge lang converision pass declarations
#ifndef EDGE_CONVERSION_PASSES
#define EDGE_CONVERISON_PASSES

include "mlir/Pass/PassBase.td"

def IntermediateEdgeLoweringPass : Pass<"Intermediate-Edge-Lowering", "mlir::ModuleOp"> {
	let summary = "First pass in the Edge lowering process to Arith and MemRef";
	let constructor = "edge::createIntermediateEdgeLoweringPass()";
	let dependentDialects = ["mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
}

def LLVMIntermediateLoweringPass : Pass<"LLVM-Intermediate-Lowering", "mlir::ModuleOp"> {
	let summary = "Lowering intermediary IR position of arith/memref/edge::OutputOp operations to LLVM.";
	let constructor = "edge::createLLVMIntermediateLoweringPass()";
	let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif
