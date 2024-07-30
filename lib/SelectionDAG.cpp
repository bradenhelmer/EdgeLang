// SelectionDAG.cpp
// ~~~~~~~~~~~~~~~~
// Implmentation of the Selction DAG interface.
#include <Edge/SelectionDAG.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <cstdio>

using namespace edge;

SelectionDAG::~SelectionDAG() { Nodes.clear(); }

void SelectionDAG::build() {
  for (const auto &I : MainBlock) {
    switch (I.getOpcode()) {
      case llvm::Instruction::Alloca:
        SelectAlloca(static_cast<const llvm::AllocaInst &>(I));
        break;
      case llvm::Instruction::Load:
        SelectLoad(static_cast<const llvm::LoadInst &>(I));
        break;
      case llvm::Instruction::Store:
        SelectStore(static_cast<const llvm::StoreInst &>(I));
        break;
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::SDiv:
        SelectBinary(I);
        break;
      default:
        break;
    }
  }
}

// Selecting an Alloca is essentially creating a stack allocation.
// This should create a 'Value' that corresponds to the pointer,
// or stack location of which the variable name is associated with.
void SelectionDAG::SelectAlloca(const llvm::AllocaInst &alloca) {
  auto size = alloca.getAllocationSize(MainBlock.getModule()->getDataLayout())
                  ->getFixedValue();
  auto N = new StackAllocSelectionDAGNode(alloca.getName(), (uint32_t)size,
                                          StackSizeInBytes);
  Nodes.push_back(N);
}

void SelectionDAG::SelectLoad(const llvm::LoadInst &load) {}
void SelectionDAG::SelectStore(const llvm::StoreInst &store) {}
void SelectionDAG::SelectBinary(const llvm::Instruction &I) {}

void SelectionDAG::printStackObjects() {
  llvm::outs() << "Stack Frame Objects\n-------------------\n";
  for (const auto &N : Nodes) {
    if (N.getType() == X86SelectionDAGNodeType::STACK_ALLOCATION) {
      const auto &SA = static_cast<const StackAllocSelectionDAGNode &>(N);
      llvm::outs() << "Name: " << SA.getName() << ", Offset: " << SA.getOffset()
                   << ", Size (Bytes): " << SA.getSizeInBytes() << "\n";
    }
  }
}
