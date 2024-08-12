// X86.h
// ~~~~~
// x86_64 target specific information.
#include <string>
#include <vector>

// System V AMD64 ABI
static const std::vector<std::string> X86_64_CALL_REGS = {"rdi", "rsi", "rdx",
                                                          "rcx", "r8",  "r9"};
