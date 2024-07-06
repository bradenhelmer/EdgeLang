// SelectionDAG.cpp
// ~~~~~~~~~~~~~~~~
// Implmentation of the Selction DAG interface.
#include <Edge/SelectionDAG.h>
#include <llvm/Support/raw_ostream.h>

using namespace edge;

void SelectionDAG::build(const llvm::BasicBlock &MainBlock) {
  for (const auto &I : MainBlock) {
    switch (I.getOpcode()) {
      case llvm::Instruction::Alloca:
      case llvm::Instruction::Add:
      case llvm::Instruction::Sub:
      case llvm::Instruction::Mul:
      case llvm::Instruction::SDiv:
      case llvm::Instruction::Call:

      default:
        break;
    }
  }
}
