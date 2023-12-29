// Toolchain.h
// ~~~~~~~~~~~
// Gathers compiler modules and executes them in order.
#include <Frontend.h>

namespace edge {

class Toolchain {
 private:
  const char *fileName;

  Lexer *lexer;
  Parser *parser;

  void initFrontend() {
    lexer = new Lexer(fileName);
    parser = new Parser(lexer);
  }

 public:
  Toolchain(const char *fileName) : fileName(fileName) { initFrontend(); }

  ~Toolchain() {
    delete lexer;
    delete parser;
  }

  void executeToolchain();

  const char *getFileName() const { return fileName; }
};

}  // namespace edge
