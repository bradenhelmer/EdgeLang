// EdgeDialect.td
// ~~~~~~~~~~~~~~
// Edge Dialect definition
#ifndef EDGE_DIALECT_TD
#define EDGE_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Edge_Dialect : Dialect {
    let name = "edge";
    let summary = "A simple dialect for arithmetic operations.";
    let cppNamespace = "::edge";
}

class EdgeOp<string mnemonic, list<Trait> traits = []> :
    Op<Edge_Dialect, mnemonic, traits>;


def ConstantOp : EdgeOp<"constant", [Pure]> {
    let summary = "constant";
    let description = [{
        Literal to SSA mlir::Value
    }];

    let arguments = (ins I64:$value);
    let results = (outs I64);
}

def AddOp : EdgeOp<"add"> {
    
    let summary = "i64 addition";
    let description = [{
        This performs basic arithmetic addition between 2 64-bit
        integers.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}

def SubOp : EdgeOp<"sub"> {
    
    let summary = "i64 subtraction";
    let description = [{
        This performs basic arithmetic subtraciton between 2 64-bit
        integers.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}

def MulOp : EdgeOp<"mul"> {
    
    let summary = "i64 multiplication";
    let description = [{
        This performs basic arithmetic multiplication between 2 64-bit
        integers.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}

def DivOp : EdgeOp<"div"> {

    let summary = "i64 division";
    let description = [{
        This performs basic arithmetic division between 2 64-bit
        integers.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];
}

def AssignRefOp : EdgeOp<"assign_ref"> {}
def AssignOp : EdgeOp<"assign"> {}
def OutputOp : EdgeOp<"output"> {}
#endif // EDGE_DIALECT_TD
