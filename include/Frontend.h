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
#include <vector>

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

static const std::unordered_map<TokenKind, const char *> tokenNames = {
#define TOKEN(X) {X, #X},
#include <TokenDef.h>
};

struct Token {
  const char *start;
  const char *end;
  const char *tokenStr = nullptr;
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

  static inline bool isWhitespace(char c) {
    return c == ' ' | c == '\t' | c == '\r';
  }
  static inline bool isIdentifierChar(char c) { return isalpha(c) || c == '_'; }
  bool lexIdentifier(Token *out, const char *currPtr);
  bool lexNumericLiteral(Token *out, const char *currPtr);
  inline void resetBufPtr() { bufPtr = sourceCode; }

 public:
  Lexer(const char *fileName)
      : fileName(fileName), sourceCode(mapSourceFile(fileName, fileLength)) {
    std::puts("Initializing Lexer...");
    bufPtr = sourceCode;
  }
  bool lexToken(Token *out);
  void lexAndPrintTokens();
};

// AST & PARSING
// ~~~~~~~~~~~~~

class ProgramAST;

class Expr {
 private:
  ProgramAST *ast;

 public:
  Expr(ProgramAST *ast) : ast(ast) {}
};

class IntegerLiteralExpr : public Expr {
 private:
  int value;

 public:
  IntegerLiteralExpr(ProgramAST *ast, int value) : Expr(ast), value(value) {}
};

class AssingeeReferenceExpr : public Expr {
 private:
  const char *assignee;

 public:
  AssingeeReferenceExpr(ProgramAST *ast, const char *assingee)
      : Expr(ast), assignee(assingee) {}
};

class BinaryOpExpr : public Expr {
 private:
  Expr *LHS;
  TokenKind op;
  Expr *RHS;

 public:
  BinaryOpExpr(ProgramAST *ast, Expr *LHS, TokenKind op, Expr *RHS)
      : Expr(ast), LHS(LHS), op(op), RHS(RHS) {}
  ~BinaryOpExpr() {
    delete LHS;
    delete RHS;
  }
};

class AssignExpr {
 private:
  const char *assignee;
  Expr *expr;

 public:
  AssignExpr(const char *assignee, Expr *expr)
      : assignee(assignee), expr(expr) {}
  ~AssignExpr() { delete expr; }
};

class ProgramAST {
 private:
  std::vector<AssignExpr> expr_list;

 public:
  ProgramAST() = default;
  ~ProgramAST() = default;
};

class Parser {
 private:
  Token *currentToken;
  Lexer *lexer;

 public:
  Parser(Lexer *lexer) : lexer(lexer), currentToken(new Token()) {
    lexer->lexToken(currentToken);
  }
  ~Parser() { delete currentToken; }
  bool parseProgram(ProgramAST *out);
};

// SEMANTIC ANALYSIS
// ~~~~~~~~~~~~~~~~~

}  // namespace edge
#endif  // EDGELANG_FRONTEND_H
