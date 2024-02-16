// LLVMIntermediateLoweringPass.cpp
// ~~~~~~~~~~
// Intermediate arith/memref lowering to LLVM
#include <Edge/Common.h>
#include <Edge/Conversion/Edge/Passes.h>
#include <Edge/Dialect/Edge/EdgeDialect.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>

using namespace mlir;
namespace edge {
#define GEN_PASS_DEF_LLVMINTERMEDIATELOWERINGPASS
#include <Edge/Conversion/Edge/Passes.h.inc>

static FlatSymbolRefAttr retrievePrintf(PatternRewriter &reWriter,
                                        ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(ctx, "printf");

  auto llvmI64Ty = IntegerType::get(ctx, 64);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI64Ty, llvmI8PtrTy, true);

  PatternRewriter::InsertionGuard insertGuard(reWriter);
  reWriter.setInsertionPointToStart(module.getBody());
  reWriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get(ctx, "printf");
}

static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                     StringRef name, StringRef value,
                                     ModuleOp module) {
  LLVM::GlobalOp global;

  if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                            LLVM::Linkage::Internal, name,
                                            builder.getStringAttr(value),
                                            /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
  Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                builder.getIndexAttr(0));
  return builder.create<LLVM::GEPOp>(
      loc,
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
      globalPtr, ArrayRef<Value>({cst0, cst0}));
}

struct OutputOpLoweringPattern : public OpConversionPattern<OutputOp> {
  using OpConversionPattern<OutputOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OutputOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &reWriter) const override {
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    FlatSymbolRefAttr printf = retrievePrintf(reWriter, parentModule);
    Location loc = op.getLoc();

    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, reWriter, "frmt_spec", StringRef("%ld\0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, reWriter, "nl", StringRef("\n\0", 2), parentModule);

    auto outputOp = cast<OutputOp>(op);

    auto callPrintFOp = reWriter.create<LLVM::CallOp>(
        reWriter.getUnknownLoc(), reWriter.getI64Type(), printf,
        llvm::ArrayRef<Value>({formatSpecifierCst, adaptor.getInput()}));

    reWriter.eraseOp(op);
    return success();
  }
};

struct LLVMIntermediateLoweringPass
    : public impl::LLVMIntermediateLoweringPassBase<
          LLVMIntermediateLoweringPass> {
  using LLVMIntermediateLoweringPassBase::LLVMIntermediateLoweringPassBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter TC(&getContext());

    RewritePatternSet patterns(&getContext());
    arith::populateArithToLLVMConversionPatterns(TC, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(TC, patterns);
    patterns.add<OutputOpLoweringPattern>(&getContext());

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createLLVMIntermediateLoweringPass() {
  return std::make_unique<LLVMIntermediateLoweringPass>();
}
}  // namespace edge
