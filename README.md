# EdgeLang
EdgeLang is a simple language supporting basic varaible creation and arithmetic operations. With its own frontend for parsing, it leverages [MLIR](https://mlir.llvm.org/) to lower IR from the Edge dialect down to LLVM IR for JIT compilation. <br>
### Warning: This project is new and for fun, it is almost certain there are bugs.

## Build Instructions
System Requirements:
-  [LLVM/Clang 17.0.6 with MLIR](https://github.com/llvm/llvm-project)
-  <strong>Linux Only</strong> (Currently uses linux sys calls for fast file reads)
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
  
    
