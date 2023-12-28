// Toolchain.h
// ~~~~~~~~~~~
// Gathers compiler modules and executes them in order.
#include <Frontend.h>

namespace edge {

class Toolchain {

private:
  const char *fileName;

  std::unique_ptr<Lexer> lexer;
  std::unique_ptr<Parser> parser;

  void initFrontend() {
    lexer = std::make_unique<Lexer>(fileName);
    parser = std::make_unique<Parser>(std::move(lexer));
  }

public:
  Toolchain(const char *fileName) : fileName(fileName) { initFrontend(); }

  void executeToolchain();

  const char *getFileName() const { return fileName; }
};

} // namespace edge
