// main.cpp
// ~~~~~~~~
// Main entrypoint for compiler.

#include <iostream>

#include <Toolchain.h>

using namespace edge;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "%s\n", "Must enter a filename for compilation.");
    exit(1);
  }

  std::unique_ptr<Toolchain> toolchain = std::make_unique<Toolchain>(argv[1]);

  return 0;
}
