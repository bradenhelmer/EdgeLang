# EdgeLang
EdgeLang is a simple language supporting basic varaible creation and arithmetic operations. With its own frontend for parsing, it leverages [MLIR](https://mlir.llvm.org/) to lower IR from the Edge dialect down to LLVM IR for JIT compilation. <br>
### Warning: This project is new and for fun, it is almost certain there are bugs.

## Build Instructions
System Requirements:
-  [LLVM/Clang with MLIR](https://github.com/llvm/llvm-project)
-  <strong>Linux</strong> (Currently uses linux sys calls for fast file reads)
-  CMake 3.22 or higher

```shell
git clone https://github.com/bradenhelmer/EdgeLang.git
cd EdgeLang
mkdir build
cd build
cmake ../
make
```
This will build an ```edge``` binary in the build directory.

## Usage
The EdgeSyntax is very simple.
- You can create variables holding integer values:
  ```
  var = 10
  ```
-  Create a varaible with an expression and then reference created variable:
    ```
    var = 10 + 20
    var_1 = var * 3
    ```
- Output an expression:
    ```
    var = 10
    output var
    ```
  ### The output statment is required for the program to compile.
- Supported operators: + - * \

## Full Example Program
```
A = 4 * 3
B = A * 5
C = A + B
output C
```
Run with:
``` edge <program_file>```<br>
Output:
```
Initializing Lexer...
Initializing Parser...
Initializing MLIR Generator...

Executing test_file.edge
Result: 72
```
## Compilation Options
1. You may specify a compilation strategy with the `-cs` option:
    - `mlir` (default): Edge code -> Edge/Func Dialects -> Arith/Func/Memref Dialects -> LLVM Dialect -> JIT compiled with MLIR execution engine.
    - `llvm`: Edge code -> LLVM IR -> JIT Compiled with LLVM LLJIT
    - `Native`: Edge code -> LLVM IR -> SelectionDAG-based Instruction Selection -> Register Allocation -> X86-64 Assembly -> ELF Binary

    All strategies utilize the same frontend methods.

    Example:<br>
    `edge -cs llvm <program>`
  
2. If you would like to output an IR or assmbly file, use the `-emit` option. This option takes no parameters but the IR file produced is based on the chosen compilation strategy:
    - `mlir` will produce one file containing three sperate modules corresponding to each stage of the lowering process:
    - `llvm` will produce an LLVM IR file.
    - `native` will produce an X86-64 Assembly file.
