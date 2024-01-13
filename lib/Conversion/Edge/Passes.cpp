// Passes.cpp
// ~~~~~~~~~~
// Edge lowering pass implementations.
#include <Edge/Conversion/Edge/Passes.h>
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;

namespace edge {
#define GEN_PASS_DEF_INTERMEDIATEEDGELOWERINGPASS
#include <Edge/Conversion/Edge/Passes.h.inc>

static MemRefType getSI64MemRefType(MLIRContext *ctx) {
  return MemRefType::get(
      {1}, IntegerType::get(ctx, CONSTANT_OP_WIDTH, IntegerType::Signed));
}

struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &reWriter) const final {
    reWriter.replaceOp(op, reWriter.create<arith::ConstantOp>(
                               reWriter.getUnknownLoc(), op.getValueAttr()));
    return success();
  }
};

struct IntermediateEdgeLoweringPass
    : public impl::IntermediateEdgeLoweringPassBase<
          IntermediateEdgeLoweringPass> {
  using IntermediateEdgeLoweringPassBase ::IntermediateEdgeLoweringPassBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    /* target.addIllegalDialect<EdgeDialect>(); */
    /* target.addDynamicallyLegalOp<OutputOp>([](OutputOp op) { */
    /*   return llvm::none_of(op->getOperandTypes(), */
    /*                        [](Type type) { return type.isa<IntegerType>();
     * }); */
    /* }); */

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createIntermediateEdgeLoweringPass() {
  return std::make_unique<IntermediateEdgeLoweringPass>();
}
}  // namespace edge
