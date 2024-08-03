// SelectionDAG.cpp
// ~~~~~~~~~~~~~~~~
// Implmentation of the Selction DAG interface.
#include <Edge/SelectionDAG.h>
#include <llvm/IR/Constants.h>
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
  StackSizeInBytes += size;
  Nodes.push_back(N);
  ValueMap.insert(std::make_pair(&alloca, SelectionDAGValue(N)));
}

void SelectionDAG::SelectLoad(const llvm::LoadInst &load) {
  // Since the language is built such that all allocas are
  // at the beginning of the 'main' basic block, we can assume
  // that all allocas have been selected and we have a valid
  // value to retrieve from the table.
  SelectionDAGValue allocaVal = ValueMap.at(load.getOperand(0));
  auto LN = new LoadSDNode(
      static_cast<StackAllocSelectionDAGNode *>(allocaVal.getNode()));
  auto U = new SelectionDAGUse(&allocaVal, LN);
  ValueMap.insert(std::make_pair(&load, SelectionDAGValue(LN)));
  LN->addOperand(U);
  Nodes.push_back(LN);
}

void SelectionDAG::SelectStore(const llvm::StoreInst &store) {
  // Storing doesn't return a value, so we can just create
  // a store node here with the pointer of choice. Constant
  // values nodes should be created here if storing an imm64.
  auto allocaVal = ValueMap.at(store.getOperand(1));
  auto SN = new StoreSDNode(
      static_cast<StackAllocSelectionDAGNode *>(allocaVal.getNode()), nullptr);
  auto ValToStore = store.getOperand(0);
  TryCreateConstantNode(ValToStore);
  auto SDValToStore = ValueMap.at(ValToStore);
  SN->addOperand(new SelectionDAGUse(&allocaVal, SN));
  SN->addOperand(new SelectionDAGUse(&SDValToStore, SN));
  Nodes.push_back(SN);
}

void SelectionDAG::SelectBinary(const llvm::Instruction &I) {
  using BOps = llvm::Instruction::BinaryOps;
  auto getX86SDNodeType = [](unsigned int Op) {
    switch (Op) {
      case BOps::Add:
        return X86SelectionDAGNodeType::ADD;
      case BOps::Sub:
        return X86SelectionDAGNodeType::SUB;
      case BOps::Mul:
        return X86SelectionDAGNodeType::MUL;
      case BOps::SDiv:
        return X86SelectionDAGNodeType::DIV;
      default:
        return X86SelectionDAGNodeType::UNKNOWN;
    }
  };

  auto RHS = I.getOperand(0);
  TryCreateConstantNode(RHS);
  auto LHS = I.getOperand(1);
  TryCreateConstantNode(LHS);

  auto N = new SelectionDAGNode(getX86SDNodeType(I.getOpcode()));
  Nodes.push_back(N);
  N->addOperand(new SelectionDAGUse(&ValueMap.at(RHS), N));
  N->addOperand(new SelectionDAGUse(&ValueMap.at(LHS), N));
  ValueMap.insert(std::make_pair(&I, SelectionDAGValue(N)));
}

void SelectionDAG::TryCreateConstantNode(llvm::Value *Val) {
  if ((ValueMap.find(Val) == ValueMap.end()) &&
      llvm::isa<llvm::ConstantInt>(Val)) {
    auto CI = static_cast<llvm::ConstantInt *>(Val);
    auto CN = new ConstantSelectionDAGNode(CI->getValue().getLimitedValue());
    Nodes.push_back(CN);
    ValueMap.insert(std::make_pair(Val, SelectionDAGValue(CN)));
  }
}

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
