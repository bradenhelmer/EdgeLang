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
  // Arithmetic Types
  ADD,
  SUB,
  MUL,
  DIV,

  // Memory Types
  STACK_ALLOCATION,
  LOAD,
  REGISTER,
  GLOBAL,
  STORE,

  // Other
  CALL,
  RET,

  // Value Types
  CONSTANT,

  // Unknown
  UNKNOWN
};

static const std::string getNodeTypeStr(X86SelectionDAGNodeType type) {
  switch (type) {
    case X86SelectionDAGNodeType::ADD:
      return "ADD";
    case X86SelectionDAGNodeType::SUB:
      return "SUB";
    case X86SelectionDAGNodeType::MUL:
      return "MUL";
    case X86SelectionDAGNodeType::DIV:
      return "DIV";
    case X86SelectionDAGNodeType::STACK_ALLOCATION:
      return "STACK_ALLOCATION";
    case X86SelectionDAGNodeType::LOAD:
      return "LOAD";
    case X86SelectionDAGNodeType::REGISTER:
      return "REGISTER";
    case X86SelectionDAGNodeType::GLOBAL:
      return "GLOBAL";
    case X86SelectionDAGNodeType::STORE:
      return "STORE";
    case X86SelectionDAGNodeType::CALL:
      return "CALL";
    case X86SelectionDAGNodeType::RET:
      return "RET";
    case X86SelectionDAGNodeType::CONSTANT:
      return "CONSTANT";
    case X86SelectionDAGNodeType::UNKNOWN:
      return "UNKNOWN";
    default:
      return "";
  };
}

static bool isValueProducing(X86SelectionDAGNodeType type) {
  switch (type) {
    case X86SelectionDAGNodeType::ADD:
    case X86SelectionDAGNodeType::SUB:
    case X86SelectionDAGNodeType::MUL:
    case X86SelectionDAGNodeType::DIV:
    case X86SelectionDAGNodeType::STACK_ALLOCATION:
    case X86SelectionDAGNodeType::LOAD:
    case X86SelectionDAGNodeType::CONSTANT:
    case X86SelectionDAGNodeType::GLOBAL:
    case X86SelectionDAGNodeType::REGISTER:
      return true;
    default:
      return false;
  }
}

class SelectionDAGNode;
class SelectionDAGValue;

// Represents an edge in the SelectionDAG between nodes.
class SelectionDAGUse {
  SelectionDAGValue *Val;
  SelectionDAGNode *User;

 public:
  SelectionDAGUse(SelectionDAGValue *Val, SelectionDAGNode *User)
      : Val(Val), User(User) {}
  ~SelectionDAGUse() = default;
  SelectionDAGValue *getValue() const { return Val; }
};

// Base node class for nodes in SelectionDAG
class SelectionDAGNode : public llvm::ilist_node<SelectionDAGNode> {
  X86SelectionDAGNodeType type;

  SelectionDAGUse *Operands[3];
  uint8_t OpCount = 0;

  SelectionDAGValue *ValueProduced = nullptr;

 public:
  SelectionDAGNode(X86SelectionDAGNodeType type) : type(type) {}
  ~SelectionDAGNode() {
    for (int i = 0; i < OpCount; ++i) delete Operands[i];
  }

  X86SelectionDAGNodeType getType() const { return type; }
  void addOperand(SelectionDAGUse *Use) { Operands[OpCount++] = Use; }
  uint8_t getOpCount() const { return OpCount; }
  SelectionDAGUse *getOperand(uint8_t index) const { return Operands[index]; }
  void setValue(SelectionDAGValue *Val) { ValueProduced = Val; }
  SelectionDAGValue *getValueProduced() const { return ValueProduced; }
};

class ConstantSelectionDAGNode : public SelectionDAGNode {
  int64_t value;

 public:
  ConstantSelectionDAGNode(int64_t value)
      : SelectionDAGNode(X86SelectionDAGNodeType::CONSTANT), value(value) {}
  int64_t getValue() const { return value; }
};

// Base class for a memory location
class MemLocSelectionDAGNode : public SelectionDAGNode {
 public:
  MemLocSelectionDAGNode(X86SelectionDAGNodeType MemType)
      : SelectionDAGNode(MemType) {}
};

// Node for stack allocations, these are built from llvm::Allocas insts.
class StackAllocSelectionDAGNode : public MemLocSelectionDAGNode {
  // Name of the variable we are referencing.
  const std::string Name;

  // Stack information.
  int32_t SizeInBytes;
  int32_t Offset;

 public:
  StackAllocSelectionDAGNode(const llvm::StringRef Name, int32_t SizeInBytes,
                             int32_t Offset)
      : MemLocSelectionDAGNode(X86SelectionDAGNodeType::STACK_ALLOCATION),
        Name(Name),
        SizeInBytes(SizeInBytes),
        Offset(Offset) {}
  const std::string &getName() const { return Name; }
  int32_t getSizeInBytes() const { return SizeInBytes; }
  int32_t getOffset() const { return Offset; }
};

// Node for register locations.
class RegSelectionDAGNode : public MemLocSelectionDAGNode {
  const std::string name;

 public:
  RegSelectionDAGNode(const std::string &name)
      : MemLocSelectionDAGNode(X86SelectionDAGNodeType::REGISTER),
        name(std::move(name)) {}

  const std::string &getName() const { return name; }
};

// Node for global locations.
class GlobalAddrSelectionDAGNode : public MemLocSelectionDAGNode {
  const std::string GlblName;

 public:
  GlobalAddrSelectionDAGNode(const std::string &name)
      : MemLocSelectionDAGNode(X86SelectionDAGNodeType::GLOBAL),
        GlblName(std::move(name)) {}
};

class LoadSDNode : public SelectionDAGNode {
  MemLocSelectionDAGNode *Dest;
  SelectionDAGValue *Src;

 public:
  // Here we initialize the destination register to nullptr as some load instrs,
  // such as lowering a call will require specific registers. Otherwise,
  // the register allocator can provide dest as it pleases.
  LoadSDNode(SelectionDAGValue *Src, MemLocSelectionDAGNode *Dest = nullptr)
      : SelectionDAGNode(X86SelectionDAGNodeType::LOAD), Src(Src), Dest(Dest) {}

  void setDest(MemLocSelectionDAGNode *Dest) const { Dest = Dest; }
};

class StoreSDNode : public SelectionDAGNode {
  MemLocSelectionDAGNode *Dest;
  SelectionDAGValue *Src;

 public:
  StoreSDNode(MemLocSelectionDAGNode *Dest, SelectionDAGValue *Src)
      : SelectionDAGNode(X86SelectionDAGNodeType::STORE),
        Dest(Dest),
        Src(Src) {}
  void setValue(SelectionDAGValue *Val) { Src = Val; }
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
  const std::string &getValueName() const { return ValueName; }
};

// Directed Acyclic Graph of Instruction Selection Nodes
class SelectionDAG {
  // Block we are selecting
  const llvm::BasicBlock &MainBlock;

  // Nodes
  llvm::ilist<SelectionDAGNode> Nodes;
  SelectionDAGNode *root;

  // Value Map
  std::unordered_map<const llvm::Value *, SelectionDAGValue *> ValueMap;
  std::vector<SelectionDAGValue *> UnMappedValues;

  // Stack allocated byte count.
  int32_t StackSizeInBytes = 0;

  // Construct the SelectionDAG
  void build();

  // Selection Methods
  void SelectAlloca(const llvm::AllocaInst &alloca);
  void SelectBinary(const llvm::Instruction &bin);
  void SelectCall(const llvm::CallInst &call);
  void SelectLoad(const llvm::LoadInst &load);
  void SelectReturn(const llvm::ReturnInst &ret);
  void SelectStore(const llvm::StoreInst &store);

  void TryCreateConstantNode(llvm::Value *Val);

 public:
  SelectionDAG(const llvm::BasicBlock &MainBlock) : MainBlock(MainBlock) {
    build();
  }

  ~SelectionDAG();

  void printRaw();

  void printStackObjects();
};

}  // namespace edge

#endif  // EDGE_SELECTION_DAG_H
