// SelectionDAG.h
// ~~~~~~~~~~~~~~
// Defines the SelectionDAG interface.
#ifndef EDGE_SELECTION_DAG_H
#define EDGE_SELECTION_DAG_H
#include <llvm/ADT/ilist.h>
#include <llvm/IR/BasicBlock.h>

#include <cstdint>

namespace edge {

enum class X86Opcode : uint32_t {
  ADD,
  SUB,
  MUL,
  DIV,
  LOAD,
  STORE,
  CALL,
  STACKALLOC
};

class SelectionDAGNode : public llvm::ilist_node<SelectionDAGNode> {
 private:
  X86Opcode Opcode;
  int32_t StackOffset;

 public:
  SelectionDAGNode(X86Opcode Opcode, int32_t StackOffset = -1)
      : Opcode(Opcode), StackOffset(StackOffset) {}
};

class SelectionDAG {
 public:
  SelectionDAG(const llvm::BasicBlock &MainBlock) { build(MainBlock); }

 private:
  void build(const llvm::BasicBlock &MainBlock);
  llvm::ilist<SelectionDAGNode> Nodes;
  llvm::DenseMap<const llvm::Value *, SelectionDAGNode *> ValueToNodeMap;

  uint32_t StackSpace = 0;
};

}  // namespace edge

#endif  // EDGE_SELECTION_DAG_H
