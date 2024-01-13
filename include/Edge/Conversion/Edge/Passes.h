// Passes.h
// ~~~~~~~~
// Edge lowering passes header.
#ifndef EDGE_CONVERSION_PASS_H
#define EDGE_CONVERSION_PASS_H
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace edge {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createIntermediateEdgeLoweringPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include <Edge/Conversion/Edge/Passes.h.inc>

}  // namespace edge

#endif  // EDGE_CONVERSION_PASS_H
