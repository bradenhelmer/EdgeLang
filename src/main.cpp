// main.cpp
// ~~~~~~~~
// Main entrypoint for compiler.

#include <Toolchain.h>

#include <iostream>

using namespace edge;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "%s\n", "Must enter a filename for compilation.");
    exit(1);
  }

  Toolchain *TC = new Toolchain(argv[1]);
  TC->executeToolchain();

  delete TC;
  return 0;
}
