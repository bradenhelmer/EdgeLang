// main.cpp
// ~~~~~~~~
// Main entrypoint for compiler.

#include <iostream>

#include <Frontend.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "%s\n", "Must enter a filename for compilation.");
    exit(1);
  }

  size_t length;
  const char *file = edge::mapSourceFile(argv[1], length);
  std::printf("%s", file);

  return 0;
}
