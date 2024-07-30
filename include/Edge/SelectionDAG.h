// SelectionDAG.h
// ~~~~~~~~~~~~~~
// Defines the SelectionDAG interface,
// as well as the various SelectionDAG node types.
#ifndef EDGE_SELECTION_DAG_H
#define EDGE_SELECTION_DAG_H
#include <llvm/ADT/ilist.h>
#include <llvm/ADT/ilist_node.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>

#include <unordered_map>

namespace edge {

enum class X86SelectionDAGNodeType : uint32_t {
  // Memory Types
  STACK_ALLOCATION,
  LOAD,
  STORE,
  // Arithmetic Types
  ADD,
  SUB,
  MUL,
  DIV,
  // Value Types
  CONSTANT,
  REGISTER
};

class SelectionDAGNode;

// Represents an edge in the SelectionDAG between nodes.
class SelectionDAGUse {
  SelectionDAGUse *Next;
  SelectionDAGUse *Prev;
  SelectionDAGNode *User;

 public:
  SelectionDAGUse() = default;
};

// Wrapper for a node producing a runtime 'value', this could be:
// 	- A pointer
// 	- Result of an instruction
class SelectionDAGValue {
  SelectionDAGNode *Node;

 public:
  SelectionDAGValue(SelectionDAGNode *Node) : Node(Node) {}
  virtual ~SelectionDAGValue() = default;
  SelectionDAGNode *getNode() const { return Node; }
};

// Base node class for nodes in SelectionDAG
class SelectionDAGNode : public llvm::ilist_node<SelectionDAGNode> {
  X86SelectionDAGNodeType type;

  SelectionDAGUse *Operands;

 public:
  SelectionDAGNode(X86SelectionDAGNodeType type) : type(type) {}
  ~SelectionDAGNode() = default;

  X86SelectionDAGNodeType getType() const { return type; }
};

class ConstantSelectionDAGNode : public SelectionDAGNode {
  int64_t value;

 public:
  ConstantSelectionDAGNode(int64_t value)
      : SelectionDAGNode(X86SelectionDAGNodeType::CONSTANT), value(value) {}
  int64_t getValue() const { return value; }
};

class StackAllocSelectionDAGNode : public SelectionDAGNode {
  // Name of the variable we are referencing.
  const std::string Name;

  // Stack information.
  int32_t SizeInBytes;
  int32_t Offset;

 public:
  StackAllocSelectionDAGNode(const llvm::StringRef Name, int32_t SizeInBytes,
                             int32_t Offset)
      : SelectionDAGNode(X86SelectionDAGNodeType::STACK_ALLOCATION),
        Name(Name),
        SizeInBytes(SizeInBytes),
        Offset(Offset) {}
  const llvm::StringRef getName() const { return Name; }
  int32_t getSizeInBytes() const { return SizeInBytes; }
  int32_t getOffset() const { return Offset; }
};

// Directed Acyclic Graph of Instruction Selection Nodes
class SelectionDAG {
  // Block we are selecting
  const llvm::BasicBlock &MainBlock;

  // Nodes
  llvm::ilist<SelectionDAGNode> Nodes;

  // Value Map
  std::unordered_map<const llvm::Value *, SelectionDAGValue> ValueMap;

  // Stack allocated byte count.
  int32_t StackSizeInBytes = 0;

  // Construct the SelectionDAG
  void build();

  // Selection Methods
  void SelectAlloca(const llvm::AllocaInst &alloca);
  void SelectLoad(const llvm::LoadInst &load);
  void SelectStore(const llvm::StoreInst &store);
  void SelectBinary(const llvm::Instruction &bin);

 public:
  SelectionDAG(const llvm::BasicBlock &MainBlock) : MainBlock(MainBlock) {
    build();
  }

  ~SelectionDAG();

  void printStackObjects();
};

}  // namespace edge

#endif  // EDGE_SELECTION_DAG_H
