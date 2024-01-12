// Passes.td
// ~~~~~~~~~
// Edge lang converision pass declarations

include "mlir/Pass/PassBase.td"

def IntermediateEdgeLoweringPass : Pass<"Intermediate-Edge-Lowering-Pass", "mlir::ModuleOp"> {
	let summary = "First pass in the Edge lowering process to Arith and MemRef";
	let constructor = "edge::createIntermediateEdgeLoweringPass()";
	let dependentDialects = ["mlir::arith::ArithDialect", "mlir::memref::MemRefDialect"];
}
