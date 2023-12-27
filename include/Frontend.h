// Frontend.h
// ~~~~~~~~~~
// Definitions for compiler frontend module:
// 1. Source Retrieval
// 2. Lexical Analysis
// 3. AST and Parsing
// 4. Semantic Analysis
// ~~~~~~~~~~~~~~~~~~~~
#ifndef EDGELANG_FRONTEND_H
#define EDGELANG_FRONTEND_H
#include <cstdint>
#include <cstring>
#include <iostream>

namespace edge {

const char *mapSourceFile(const char *fileName, size_t &length);
static void handleSourceReadError(const char *msg, uint8_t errCode,
                                  const char *fileName = "") {
  fprintf(stderr, "%s: %s\n", fileName, msg);
  exit(errCode);
}

} // namespace edge
#endif // EDGELANG_FRONTEND_H
