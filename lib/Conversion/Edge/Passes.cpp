// Passes.cpp
// ~~~~~~~~~~
// Edge lowering pass implementations.
#include <Edge/Conversion/Edge/Passes.h>
#include <Edge/Dialect/Edge/EdgeDialect.h>

#include <iostream>
using namespace mlir;

namespace edge {
#define GEN_PASS_DEF_INTERMEDIATEEDGELOWERINGPASS
#include <Edge/Conversion/Edge/Passes.h.inc>

static MemRefType getSI64MemRefType(MLIRContext *ctx) {
  return MemRefType::get(
      {1}, IntegerType::get(ctx, CONSTANT_OP_WIDTH, IntegerType::Signed));
}

EdgeTypeConverter::EdgeTypeConverter(mlir::MLIRContext *ctx) : ctx(ctx) {
  addConversion([&](IntegerType type) { return convertIntegerType(type); });
}

Type EdgeTypeConverter::convertIntegerType(IntegerType type) {
  return IntegerType::get(getContext(), CONSTANT_OP_WIDTH,
                          IntegerType::Signless);
}

struct ConstantOpLoweringPattern : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ConstantOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    TypeConverter *TC = getTypeConverter();
    mlir::Type newType = TC->convertType(op.getType());
    if (!newType) return failure();
    auto newAttr = IntegerAttr::get(newType, op.getValue());
    auto newOp =
        reWriter.create<arith::ConstantOp>(reWriter.getUnknownLoc(), newAttr);
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }
};

struct AddOpLoweringPattern : public OpConversionPattern<AddOp> {
  using OpConversionPattern<AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    auto newOp = reWriter.create<arith::AddIOp>(
        reWriter.getUnknownLoc(), adaptor.getLhs(), adaptor.getRhs());
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }
};

struct SubOpLoweringPattern : public OpConversionPattern<SubOp> {
  using OpConversionPattern<SubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    auto newOp = reWriter.create<arith::SubIOp>(
        reWriter.getUnknownLoc(), adaptor.getLhs(), adaptor.getRhs());
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }
};

struct MulOpLoweringPattern : public OpConversionPattern<MulOp> {
  using OpConversionPattern<MulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    auto newOp = reWriter.create<arith::MulIOp>(
        reWriter.getUnknownLoc(), adaptor.getLhs(), adaptor.getRhs());
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }
};

struct DivOpLoweringPattern : public OpConversionPattern<DivOp> {
  using OpConversionPattern<DivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DivOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    auto newOp = reWriter.create<arith::DivSIOp>(
        reWriter.getUnknownLoc(), adaptor.getLhs(), adaptor.getRhs());
    op.replaceAllUsesWith(newOp.getResult());
    op.erase();
    return success();
  }
};

struct OutputOpLoweringPattern : public OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    reWriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct IntermediateEdgeLoweringPass
    : public impl::IntermediateEdgeLoweringPassBase<
          IntermediateEdgeLoweringPass> {
  using IntermediateEdgeLoweringPassBase ::IntermediateEdgeLoweringPassBase;

 private:
  void populateEdgeConversionPatterns(RewritePatternSet &patterns,
                                      EdgeTypeConverter &converter) {
    patterns
        .add<ConstantOpLoweringPattern, AddOpLoweringPattern,
             SubOpLoweringPattern, MulOpLoweringPattern, DivOpLoweringPattern>(
            converter, &getContext());
  }

 public:
  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();

    // Marking op legality
    target.addIllegalDialect<EdgeDialect>();
    target.addLegalOp<OutputOp>();

    RewritePatternSet patterns(&getContext());
    EdgeTypeConverter converter(&getContext());
    populateEdgeConversionPatterns(patterns, converter);
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
