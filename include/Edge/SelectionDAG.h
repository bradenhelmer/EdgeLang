// SelectionDAG.h
// ~~~~~~~~~~~~~~
// Defines the SelectionDAG interface.
#ifndef EDGE_SELECTION_DAG_H
#define EDGE_SELECTION_DAG_H
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Value.h>

namespace edge {

enum class SDNodeType : unsigned char {
  ADD,
  SUB,
  MUL,
  DIV,
  LOAD,
  STORE,
  CALL,
  RET
};

class SelectionDAGNode {
 private:
  SDNodeType type;

 public:
  SDNodeType getNodeType() const { return type; }
};

// This is a simple SelectionDAG implementation. It is loosely
// based off of LLVMs SD, using an llvm::DenseMap to map
// llvm::Values to edge::SelectionDAGNodes.
class SelectionDAG {
 public:
  SelectionDAG(const llvm::BasicBlock &MainBlock) { build(MainBlock); }

 private:
  llvm::DenseMap<const llvm::Value *, SelectionDAGNode> Nodes;

  void build(const llvm::BasicBlock &MainBlock);
};

}  // namespace edge

#endif  // EDGE_SELECTION_DAG_H
