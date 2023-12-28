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
#include <memory>

namespace edge {

// SOURCE RETRIEVAL
// ~~~~~~~~~~~~~~~~

const char *mapSourceFile(const char *fileName, size_t &length);
static void handleSourceReadError(const char *msg, uint8_t errCode,
                                  const char *fileName = "") {
  fprintf(stderr, "%s: %s\n", fileName, msg);
  exit(errCode);
}

// LEXICAL ANALYSIS
// ~~~~~~~~~~~~~~~~

class Lexer {
private:
  const char *fileName;
  size_t fileLength;
  const char *sourceCode;

public:
  Lexer(const char *fileName)
      : fileName(fileName), sourceCode(mapSourceFile(fileName, fileLength)) {
    std::printf("Lexer initialized with source code:\n%s", sourceCode);
  }
};

// AST & PARSING
// ~~~~~~~~~~~~~

class Parser {
private:
  std::unique_ptr<Lexer> lexer;

public:
  Parser(std::unique_ptr<Lexer> lexer) : lexer(std::move(lexer)) {}
};

// SEMANTIC ANALYSIS
// ~~~~~~~~~~~~~~~~~

} // namespace edge
#endif // EDGELANG_FRONTEND_H
