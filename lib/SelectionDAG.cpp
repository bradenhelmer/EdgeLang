// SelectionDAG.cpp
// ~~~~~~~~~~~~~~~~
// Implmentation of the Selction DAG interface.
#include <Edge/SelectionDAG.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

using namespace edge;

void SelectionDAG::build(const llvm::BasicBlock &MainBlock) {
  for (const auto &I : MainBlock) {
    switch (I.getOpcode()) {
      case llvm::Instruction::Alloca: {
        auto AllocaNode =
            new SelectionDAGNode(X86Opcode::STACKALLOC, StackSpace);
        StackSpace += static_cast<const llvm::AllocaInst &>(I)
                          .getAllocationSize(MainBlock.getDataLayout())
                          ->getFixedValue();
        Nodes.addNodeToList(AllocaNode);
        ValueToNodeMap.try_emplace(&I, AllocaNode);
        break;
      }
      case llvm::Instruction::Load: {
        auto LoadNode = new SelectionDAGNode(X86Opcode::LOAD);
      }
    }
  }
}
