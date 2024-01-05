// EdgeDialect.cpp
// ~~~~~~~~~~~~~~~
// Implementation of the Edge dialect
#include <Edge/Dialect/Edge/EdgeDialect.h>

namespace mlir::edge {

void EdgeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <Edge/Dialect/Edge/EdgeOps.cpp.inc>
      >();
}
}  // namespace mlir::edge
