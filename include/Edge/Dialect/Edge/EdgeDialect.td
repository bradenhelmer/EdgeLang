// EdgeDialect.td
// ~~~~~~~~~~~~~~
// Edge Dialect definition
#ifndef EDGE_DIALECT_TD
#define EDGE_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpBase.td"

def Edge_Dialect : Dialect {
    let name = "edge";
    let summary = "A simple dialect for arithmetic operations.";
    let cppNamespace = "::edge";
}

class EdgeOp<string mnemonic, list<Trait> traits = []> :
    Op<Edge_Dialect, mnemonic, traits>;


def ConstantOp : EdgeOp<"constant", [Pure, ConstantLike]> {
    let summary = "constant";
    let description = [{
        Represent a constant, signed 64-bit integer.
    }];

    let arguments = (ins SI64Attr:$value);
    let results = (outs SI64);

  	let builders = [
		OpBuilder<(ins "int64_t":$value)>
  	];

	let hasVerifier = 1;
	let hasFolder = 1;
}

def AddOp : EdgeOp<"add"> {
    
    let summary = "i64 addition";
    let description = [{
        This performs basic arithmetic addition between 2 64-bit
        integers.
    }];
    let arguments = (ins SI64:$lhs, SI64:$rhs);
    let results = (outs SI64);

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
    let arguments = (ins SI64:$lhs, SI64:$rhs);
    let results = (outs SI64);

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
    let arguments = (ins SI64:$lhs, SI64:$rhs);
    let results = (outs SI64);

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
    let arguments = (ins SI64:$lhs, SI64:$rhs);
    let results = (outs SI64);

    let builders = [
        OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
    ];

	let hasCanonicalizeMethod = 1;
}

def RefOp : EdgeOp<"assign_ref"> {
	let summary = "Reference to a symbol that has been assinged a value.";
	let description = [{
		Reference to a symbol that holds a value.	
	}];
	let arguments = (ins SymbolRefAttr:$symbol);
	let results = (outs SI64);

	let builders = [
		OpBuilder<(ins "llvm::StringRef":$symbol)>
	];
}

def AssignOp : EdgeOp<"assign"> {
	let summary = "Operation to assign a value or expression to a symbol.";
	let description = [{
		Assign a value to a symbol e.g 
		A = 3
		B = A + 1
	}];
	let arguments = (ins SymbolNameAttr:$symbol, AnyInteger:$value);
}

def OutputOp : EdgeOp<"output"> {
	let summary = "Output an expression or value";
	let arguments = (ins AnyInteger:$output_val);
}

#endif // EDGE_DIALECT_TD
