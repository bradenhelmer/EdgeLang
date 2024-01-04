// EdgeDialect.td
// ~~~~~~~~~~~~~~
// Edge Dialect definition
#ifndef EDGE_DIALECT_TD
#define EDGE_DIALECT_TD

include "mlir/IR/DialectBase.td"

def Edge_Dialect : Dialect {
    let name = "edge";
    let summary = "A simple dialect for arithmetic operations.";
    let cppNamespace = "edge";
}

#endif // EDGE_DIALECT_TD
