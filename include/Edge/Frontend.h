// Frontend.h
// ~~~~~~~~~~
// Definitions for compiler frontend module:
// 1. Source Retrieval
// 2. Lexical Analysis
// 3. AST and Parsing
// ~~~~~~~~~~~~~~~~~~~~
#ifndef EDGELANG_FRONTEND_H
#define EDGELANG_FRONTEND_H
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <cstring>
#include <iostream>
#include <map>
#include <optional>

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

enum TokenKind : uint8_t {
#define TOKEN(X) X,
#include "TokenDef.h"
  NUM_TOKENS
};

static const std::map<llvm::StringRef, TokenKind> keywords = {
#define KEYWORD(X) {#X, keyword_##X},
#include "TokenDef.h"
};

static const std::map<TokenKind, const char *> tokenNames = {
#define TOKEN(X) {X, #X},
#include "TokenDef.h"
};

static inline bool isOperator(TokenKind kind) {
  return kind > OPERATOR_START && kind < OPERATOR_END;
}

struct Token {
  const char *start;
  const char *end;
  std::string tokenStr;
  TokenKind kind;
  Token() : tokenStr("") {}
  inline size_t getLength() const { return (end - start) + 1; }
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
    bufPtr = sourceCode;
  }
  bool lexToken(Token *out);
  void lexAndPrintTokens();
  void printCurrChar() const { std::printf("%c\n", *bufPtr); }
};

// AST & PARSING
// ~~~~~~~~~~~~~

class ProgramAST;

class Expr {
 private:
  ProgramAST *ast;

 public:
  Expr(ProgramAST *ast) : ast(ast) {}
  virtual ~Expr() = default;
  enum ExprType : uint8_t { EXPR, INTEGER_LITERAL, ASSIGNEE_REF, BINOP };
  virtual ExprType getType() const { return EXPR; }
};

class IntegerLiteralExpr : public Expr {
 private:
  int64_t value;

 public:
  IntegerLiteralExpr(ProgramAST *ast, int64_t value)
      : Expr(ast), value(value) {}
  int64_t getValue() const { return value; }
  ExprType getType() const override { return INTEGER_LITERAL; }
};

class AssigneeReferenceExpr : public Expr {
 private:
  const std::string assignee;

 public:
  AssigneeReferenceExpr(ProgramAST *ast, const std::string &assignee)
      : Expr(ast), assignee(assignee) {}
  llvm::StringRef getAssignee() const { return assignee; }
  ExprType getType() const override { return ASSIGNEE_REF; }
};

class BinaryOpExpr : public Expr {
 private:
  Expr *LHS;
  TokenKind op;
  Expr *RHS;

 public:
  BinaryOpExpr(ProgramAST *ast, Expr *LHS, TokenKind op, Expr *RHS)
      : Expr(ast), LHS(std::move(LHS)), op(op), RHS(std::move(RHS)) {}
  ~BinaryOpExpr() override {
    delete LHS;
    delete RHS;
  }
  ExprType getType() const override { return BINOP; }
  TokenKind getOp() const { return op; }
  Expr &getLHS() const { return *LHS; }
  Expr &getRHS() const { return *RHS; }
};

class AssignStmt {
 private:
  ProgramAST *ast;
  const std::string assignee;
  Expr *expr;

 public:
  AssignStmt(ProgramAST *ast, const std::string &assignee, Expr *expr)
      : ast(ast), assignee(assignee), expr(std::move(expr)) {}
  ~AssignStmt() { delete expr; }
  llvm::StringRef getAssignee() const { return assignee; }
  Expr &getExpr() const { return *expr; }
};

class OutputStmt {
 private:
  ProgramAST *ast;
  Expr *expr;

 public:
  OutputStmt(ProgramAST *ast, Expr *expr) : ast(ast), expr(std::move(expr)) {}
  ~OutputStmt() { delete expr; }
  Expr &getExpr() const { return *expr; }
};

class ProgramAST {
 private:
  llvm::SmallVector<AssignStmt *> exprList;
  OutputStmt *output;

 public:
  ProgramAST() = default;
  ~ProgramAST();
  void attachAssignExpr(AssignStmt *assignExpr) {
    exprList.push_back(assignExpr);
  }
  void attachOutputStmt(OutputStmt *stmt) { output = stmt; }
  llvm::SmallVector<AssignStmt *> &getAssignStmts() { return exprList; }
  OutputStmt &getOutputStmt() { return *output; }
};

enum Precedence : uint8_t {
  error = 0,
  base = 1,
  additive = 2,
  multiplicative = 3
};

inline static Precedence getOperatorPrecedence(TokenKind kind) {
  switch (kind) {
    case ADD:
    case SUB:
      return additive;
    case MUL:
    case DIV:
      return multiplicative;
    default:
      return error;
  }
}

class Parser {
 private:
  Token *currentToken;
  Lexer *lexer;

  static bool parsingError(const char *msg) {
    fprintf(stderr, "Parsing Error: %s\n", msg);
    return false;
  }

  Expr *parseExpr(ProgramAST *out);
  Expr *parseBinaryOpExpr(ProgramAST *out, Expr *LHS, Precedence prec);
  inline bool match(TokenKind desired) const {
    return currentToken->kind == desired;
  }
  inline void advance() const { lexer->lexToken(currentToken); }

 public:
  Parser(Lexer *lexer) : lexer(lexer), currentToken(new Token()) { advance(); }
  ~Parser() { delete currentToken; }
  bool parseProgram(ProgramAST *out);
};

}  // namespace edge
#endif  // EDGELANG_FRONTEND_H
