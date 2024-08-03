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

#include <sstream>
#include <unordered_map>
#include <vector>

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
  REGISTER,
  // Unknown
  UNKNOWN
};

class SelectionDAGNode;
class SelectionDAGValue;
class StackAllocSelectionDAGNode;

// Represents an edge in the SelectionDAG between nodes.
class SelectionDAGUse {
  SelectionDAGValue *Val;
  SelectionDAGNode *User;

 public:
  SelectionDAGUse(SelectionDAGValue *Val, SelectionDAGNode *User)
      : Val(Val), User(User) {}
  ~SelectionDAGUse() = default;
};

// Base node class for nodes in SelectionDAG
class SelectionDAGNode : public llvm::ilist_node<SelectionDAGNode> {
  X86SelectionDAGNodeType type;

  SelectionDAGUse *Operands[3];
  uint8_t OpCount = 0;

 public:
  SelectionDAGNode(X86SelectionDAGNodeType type) : type(type) {}
  ~SelectionDAGNode() {
    for (int i = 0; i < OpCount; ++i) delete Operands[i];
  }

  X86SelectionDAGNodeType getType() const { return type; }
  void addOperand(SelectionDAGUse *Use) { Operands[OpCount++] = Use; }
  SelectionDAGUse *getOperand(uint8_t index) { return Operands[index]; }
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
  const std::string &getName() const { return Name; }
  int32_t getSizeInBytes() const { return SizeInBytes; }
  int32_t getOffset() const { return Offset; }
};

class LoadSDNode : public SelectionDAGNode {
  StackAllocSelectionDAGNode *MemLoc;

 public:
  LoadSDNode(StackAllocSelectionDAGNode *MemLoc)
      : SelectionDAGNode(X86SelectionDAGNodeType::LOAD), MemLoc(MemLoc) {};
};
class StoreSDNode : public SelectionDAGNode {
  StackAllocSelectionDAGNode *MemLoc;
  SelectionDAGValue *ValToStore;

 public:
  StoreSDNode(StackAllocSelectionDAGNode *MemLoc, SelectionDAGValue *ValToStore)
      : SelectionDAGNode(X86SelectionDAGNodeType::STORE),
        MemLoc(MemLoc),
        ValToStore(ValToStore) {}
  void setValue(SelectionDAGValue *Val) { ValToStore = Val; }
};

// Wrapper for a node producing a runtime 'value', this could be:
// 	- A pointer
// 	- Result of an instruction
class SelectionDAGValue {
  SelectionDAGNode *Node;
  std::string ValueName;

  inline static unsigned short count = 0;

  std::string constructValueName() {
    if (Node->getType() == X86SelectionDAGNodeType::STACK_ALLOCATION) {
      return static_cast<StackAllocSelectionDAGNode *>(Node)->getName();
    } else {
      std::stringstream ss;
      ss << "%" << count++;
      return ss.str();
    }
  }

 public:
  SelectionDAGValue(SelectionDAGNode *Node)
      : Node(Node), ValueName(constructValueName()) {}

  ~SelectionDAGValue() = default;
  SelectionDAGNode *getNode() const { return Node; }
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

  void TryCreateConstantNode(llvm::Value *Val);

 public:
  SelectionDAG(const llvm::BasicBlock &MainBlock) : MainBlock(MainBlock) {
    build();
  }

  ~SelectionDAG();

  void printStackObjects();
};

}  // namespace edge

#endif  // EDGE_SELECTION_DAG_H
