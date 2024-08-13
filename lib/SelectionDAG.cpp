// SelectionDAG.cpp
// ~~~~~~~~~~~~~~~~
// Implmentation of the Selction DAG interface.
#include <Edge/SelectionDAG.h>
#include <Edge/X86.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <cstdio>

using namespace edge;

SelectionDAG::~SelectionDAG() {
  Nodes.clear();
  for (auto &P : ValueMap) {
    delete P.second;
  }
  for (auto &P : UnMappedValues) {
    delete P;
  }
}

void SelectionDAG::build() {
  for (const auto &I : MainBlock) {
    switch (I.getOpcode()) {
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::SDiv:
        SelectBinary(I);
        break;
      case llvm::Instruction::Alloca:
        SelectAlloca(static_cast<const llvm::AllocaInst &>(I));
        break;
      case llvm::Instruction::Call:
        SelectCall(static_cast<const llvm::CallInst &>(I));
        break;
      case llvm::Instruction::Load:
        SelectLoad(static_cast<const llvm::LoadInst &>(I));
        break;
      case llvm::Instruction::Ret:
        SelectReturn(static_cast<const llvm::ReturnInst &>(I));
        break;
      case llvm::Instruction::Store:
        SelectStore(static_cast<const llvm::StoreInst &>(I));
        break;
      default:
        break;
    }
  }
  root = &Nodes.back();
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
  auto V = new SelectionDAGValue(N);
  ValueMap.insert(std::make_pair(&alloca, V));
  N->setValue(V);
}

void SelectionDAG::SelectBinary(const llvm::Instruction &I) {
  auto getX86SDNodeType = [](unsigned int Op) {
    using BOps = llvm::Instruction::BinaryOps;
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
  N->addOperand(new SelectionDAGUse(ValueMap.at(RHS), N));
  N->addOperand(new SelectionDAGUse(ValueMap.at(LHS), N));
  auto V = new SelectionDAGValue(N);
  ValueMap.insert(std::make_pair(&I, V));
  N->setValue(V);
}

// Since EdgeLang only has one call to printf at the end,
// this function can be specialized for that.
void SelectionDAG::SelectCall(const llvm::CallInst &call) {
  // First get the global format string as a usable value.
  auto FMT_SPEC_G = MainBlock.getModule()->getNamedGlobal("fmt_spec");
  auto GAN = new GlobalAddrSelectionDAGNode("fmt_spec");
  Nodes.push_back(GAN);
  auto GAV = new SelectionDAGValue(GAN);
  ValueMap.insert(std::make_pair(FMT_SPEC_G, GAV));
  GAN->setValue(GAV);

  // Create initial call node
  auto CN = new SelectionDAGNode(X86SelectionDAGNodeType::CALL);

  // Now iterate over parameters and create load nodes coressponding
  // to X86 Sys-V calling conv.
  auto parameters = call.getOperandList();
  for (int i = 0; i < call.getNumOperands() - 1; ++i) {
    // Create load, these shouldn't create values in the map as they don't
    // correspond to existing llvm values.
    const auto &param = parameters[i];
    auto PV = ValueMap.at(param);
    auto RN = new RegSelectionDAGNode(X86_64_CALL_REGS[i]);
    Nodes.push_back(RN);
    auto RV = new SelectionDAGValue(RN);
    RN->setValue(RV);
    UnMappedValues.push_back(RV);
    auto LN = new LoadSDNode(PV, RN);
    Nodes.push_back(LN);
    LN->addOperand(new SelectionDAGUse(PV, LN));
    LN->addOperand(new SelectionDAGUse(RV, LN));
    auto LV = new SelectionDAGValue(LN);
    UnMappedValues.push_back(LV);
    LN->setValue(LV);
    CN->addOperand(new SelectionDAGUse(LV, CN));
  }

  Nodes.push_back(CN);
}

void SelectionDAG::SelectLoad(const llvm::LoadInst &load) {
  // Since the language is built such that all allocas are
  // at the beginning of the 'main' basic block, we can assume
  // that all allocas have been selected and we have a valid
  // value to retrieve from the table.
  auto allocaVal = ValueMap.at(load.getOperand(0));
  auto LN = new LoadSDNode(allocaVal);
  auto U = new SelectionDAGUse(allocaVal, LN);
  auto LV = new SelectionDAGValue(LN);
  ValueMap.insert(std::make_pair(&load, LV));
  LN->addOperand(U);
  LN->setValue(LV);
  Nodes.push_back(LN);
}

// Returns also do not create values.
void SelectionDAG::SelectReturn(const llvm::ReturnInst &ret) {
  auto ValToRet = ret.getOperand(0);
  TryCreateConstantNode(ValToRet);
  auto RN = new SelectionDAGNode(X86SelectionDAGNodeType::RET);
  RN->addOperand(new SelectionDAGUse(ValueMap.at(ValToRet), RN));
  Nodes.push_back(RN);
}

void SelectionDAG::SelectStore(const llvm::StoreInst &store) {
  // Storing doesn't return a value, so we can just create
  // a store node here with the pointer of choice. Constant
  // values nodes should be created here if storing an imm64.
  auto ValToStore = store.getOperand(0);
  auto allocaVal = ValueMap.at(store.getOperand(1));
  auto SN = new StoreSDNode(
      static_cast<StackAllocSelectionDAGNode *>(allocaVal->getNode()), nullptr);
  TryCreateConstantNode(ValToStore);
  auto SDValToStore = ValueMap.at(ValToStore);
  SN->addOperand(new SelectionDAGUse(allocaVal, SN));
  SN->addOperand(new SelectionDAGUse(SDValToStore, SN));
  Nodes.push_back(SN);
}

void SelectionDAG::TryCreateConstantNode(llvm::Value *Val) {
  if ((ValueMap.find(Val) == ValueMap.end()) &&
      llvm::isa<llvm::ConstantInt>(Val)) {
    auto CI = static_cast<llvm::ConstantInt *>(Val);
    auto CN = new ConstantSelectionDAGNode(CI->getValue().getLimitedValue());
    Nodes.push_back(CN);
    auto CV = new SelectionDAGValue(CN);
    ValueMap.insert(std::make_pair(Val, CV));
    CN->setValue(CV);
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

void SelectionDAG::printRaw() {
  uint16_t idx = 0;
  llvm::outs() << "DAG\n---\n";
  for (const auto &N : Nodes) {
    auto type = N.getType();
    llvm::outs() << "Node #" << idx++ << ": " << getNodeTypeStr(type);
    if (isValueProducing(type)) {
      llvm::outs() << " -> " << N.getValueProduced()->getValueName();
    }
    llvm::outs() << '\n';

    for (int i = 0; i < N.getOpCount(); ++i) {
      llvm::outs() << "  |_ " << N.getOperand(i)->getValue()->getValueName()
                   << "\n";
    }
  }
}
