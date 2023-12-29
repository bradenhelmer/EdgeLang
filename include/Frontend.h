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
#include <unordered_map>

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
enum TokenKind : unsigned short {
#define TOKEN(X) X,
#include <TokenDef.h>
  NUM_TOKENS
};

static const std::unordered_map<const char *, TokenKind> keywords = {
#define KEYWORD(X) {#X, keyword_##X},
#include <TokenDef.h>
};

struct Token {
  const char *start;
  const char *end;
  TokenKind kind;
  Token() = default;
  size_t getLength() const { return (end - start) + 1; }
};

class Lexer {
 private:
  const char *fileName;
  size_t fileLength;
  const char *sourceCode;
  const char *bufPtr;

  static bool isWhitespace(char c) { return c == ' ' | c == '\t' | c == '\r'; }
  static bool isIdentifierChar(char c) { return isalpha(c) || c == '_'; }

 public:
  Lexer(const char *fileName)
      : fileName(fileName), sourceCode(mapSourceFile(fileName, fileLength)) {
    std::puts("Initializing Lexer...");
    bufPtr = sourceCode;
  }
  bool lexToken(Token *out);
  bool lexIdentifier(Token *out, const char *currPtr);
  bool lexNumericLiteral(Token *out, const char *currPtr);
  void lexAndPrintTokens();
};

// AST & PARSING
// ~~~~~~~~~~~~~

class Parser {
 private:
  Token *currentToken;
  Lexer *lexer;

 public:
  Parser(Lexer *lexer) : lexer(lexer), currentToken(new Token()) {}
  ~Parser() { delete currentToken; }
};

// SEMANTIC ANALYSIS
// ~~~~~~~~~~~~~~~~~

}  // namespace edge
#endif  // EDGELANG_FRONTEND_H
