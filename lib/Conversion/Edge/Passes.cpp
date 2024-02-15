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

static MemRefType getI64MemRefType(MLIRContext *ctx) {
  return MemRefType::get(
      {1}, IntegerType::get(ctx, CONSTANT_OP_WIDTH, IntegerType::Signless));
}

static arith::ConstantIndexOp zerothIdx = nullptr;

static arith::ConstantIndexOp &getZeroth(OpBuilder &builder) {
  if (!zerothIdx)
    zerothIdx =
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
  return zerothIdx;
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

struct AssignOpLoweringPattern : public OpConversionPattern<AssignOp> {
  EdgeSymbolTable *symTable;

  AssignOpLoweringPattern(MLIRContext *context, EdgeSymbolTable *symTable)
      : OpConversionPattern<AssignOp>(context), symTable(symTable) {}

  LogicalResult matchAndRewrite(
      AssignOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    // Create allocation if not already
    if (symTable->find(op.getSymbol()) == symTable->end()) {
      //
      // Allocate at the top
      Block *currBlock = reWriter.getBlock();
      reWriter.setInsertionPointToStart(currBlock);
      // Create allocation and insert into symbol table
      memref::AllocOp allocation = reWriter.create<memref::AllocOp>(
          reWriter.getUnknownLoc(), getI64MemRefType(getContext()));
      symTable->insert(
          std::pair<llvm::StringRef, Value>(op.getSymbol(), allocation));
    }

    Value valToStore = adaptor.getValue();
    Value allocatedMem = symTable->find(op.getSymbol())->second;

    reWriter.create<memref::StoreOp>(reWriter.getUnknownLoc(), valToStore,
                                     allocatedMem,
                                     getZeroth(reWriter).getResult());
    op.erase();

    return success();
  }
};

struct RefOpLoweringPattern : public OpConversionPattern<RefOp> {
  EdgeSymbolTable *symTable;

  RefOpLoweringPattern(MLIRContext *context, EdgeSymbolTable *symTable)
      : OpConversionPattern<RefOp>(context), symTable(symTable) {}

  LogicalResult matchAndRewrite(
      RefOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    llvm::StringRef refSym = op.getSymbol().getRootReference().getValue();
    if (symTable->find(refSym) == symTable->end()) {
      return emitError(op.getLoc(),
                       "Cannot reference symbol not in symbol table!");
    }
    memref::LoadOp load = reWriter.create<memref::LoadOp>(
        reWriter.getUnknownLoc(), symTable->at(refSym),
        getZeroth(reWriter).getResult());
    ;
    op.replaceAllUsesWith(load.getResult());
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
                                      EdgeTypeConverter &converter,
                                      EdgeSymbolTable &symTable) {
    patterns
        .add<ConstantOpLoweringPattern, AddOpLoweringPattern,
             SubOpLoweringPattern, MulOpLoweringPattern, DivOpLoweringPattern>(
            converter, &getContext())
        .add<AssignOpLoweringPattern, RefOpLoweringPattern>(&getContext(),
                                                            &symTable);
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
    EdgeSymbolTable symTable;
    populateEdgeConversionPatterns(patterns, converter, symTable);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createIntermediateEdgeLoweringPass() {
  return std::make_unique<IntermediateEdgeLoweringPass>();
};
}  // namespace edge
