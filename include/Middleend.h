// Middleend.h
// ~~~~~~~~~~~
// Definitions for the middle end of the compiler:
// 1. Edge Dialect
#include <mlir/IR/Dialect.h>

class EdgeDialect : public mlir::Dialect {
 public:
  explicit EdgeDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "edge"; }

  void initialize();
};
