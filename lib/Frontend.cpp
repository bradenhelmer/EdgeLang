// Frontend.cpp
// ~~~~~~~~~~~~
// Implementations of frontend module
#include <Edge/Common.h>
#include <Edge/Frontend.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

namespace edge {

// Fast file read with mmap
const char *mapSourceFile(const char *fileName, size_t &length) {
  int fd = open(fileName, O_RDONLY);
  if (fd == -1) {
    handleSourceReadError("Error opening file.", ENOENT);
  }
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    handleSourceReadError("Error with fstat obtaining filesize.", EIO);
  }

  length = sb.st_size;

  madvise(NULL, length, MADV_SEQUENTIAL);
  const char *addr = static_cast<const char *>(
      mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0u));

  if (addr == MAP_FAILED) {
    handleSourceReadError("Error with mmap.", EIO);
  }
  return addr;
}

// LEXER IMPLEMENTATIONS

bool Lexer::lexToken(Token *out) {
  const char *currPtr = bufPtr;

  if (isWhitespace(*currPtr)) {
    do {
      currPtr++;
    } while (isWhitespace(*currPtr));
  }

  out->start = currPtr;
  out->end = currPtr;

  switch (*currPtr) {
    case 0:
      out->kind = EDGEEOF;
      return true;
    case '\n':
      out->kind = NEWLINE;
      break;
      // clang-format off
    case 'A': case 'B': case 'C': case 'D': case 'E': case 'F': case 'G':
    case 'H': case 'I': case 'J': case 'K': case 'L': case 'M': case 'N':
    case 'O': case 'P': case 'Q': case 'R': case 'S': case 'T': case 'U':
    case 'V': case 'W': case 'X': case 'Y': case 'Z':
    case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
    case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
    case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
    case 'v': case 'w': case 'x': case 'y': case 'z':
    case '_':
      return lexIdentifier(out, currPtr);
    case '0': case '1': case '2': case '3': case '4':
    case '5': case '6': case '7': case '8': case '9':
      return lexNumericLiteral(out, currPtr);
      // clang-format on
    case '=':
      out->kind = ASSIGN;
      break;
    case '+':
      out->kind = ADD;
      break;
    case '-':
      out->kind = SUB;
      break;
    case '*':
      out->kind = MUL;
      break;
    case '/':
      out->kind = DIV;
      break;
    case '(':
      out->kind = RPAREN;
      break;
    case ')':
      out->kind = RPAREN;
      break;
    default:
      return false;
  }
  currPtr++;
  bufPtr = currPtr;
  return true;
}

bool Lexer::lexIdentifier(Token *out, const char *currPtr) {
  out->kind = ID;
  do {
    currPtr++;
  } while (isIdentifierChar(*currPtr));
  out->end = currPtr - 1;

  out->tokenStr = llvm::StringRef(out->start, out->getLength());

  if (keywords.find(out->tokenStr) != keywords.end()) {
    out->kind = keywords.at(out->tokenStr);
  }

  bufPtr = currPtr;
  return true;
}

bool Lexer::lexNumericLiteral(Token *out, const char *currPtr) {
  out->kind = INTEGER;
  do {
    currPtr++;
  } while (isdigit(*currPtr));
  out->end = currPtr - 1;

  out->tokenStr = llvm::StringRef(out->start, out->getLength());

  bufPtr = currPtr;
  return true;
}

void Lexer::lexAndPrintTokens() {
  resetBufPtr();
  Token tok;
  while (lexToken(&tok) && tok.kind != EDGEEOF) {
    std::printf("%s\n", tokenNames.at(tok.kind));
  }
  resetBufPtr();
}

// PARSER/AST IMPLEMENTATIONS
ProgramAST::~ProgramAST() {
  std::for_each(exprList.begin(), exprList.end(),
                [](const AssignStmt *AE) { delete AE; });
  delete output;
}

bool Parser::parseProgram(ProgramAST *out) {
  if (!match(ID) && !match(keyword_output)) {
    return parsingError(
        "Program must start with an identifier or output statment!");
  }

  while (!match(keyword_output) && match(ID)) {
    std::string assignee(currentToken->tokenStr);
    advance();

    if (!match(ASSIGN)) {
      return parsingError(
          "Identifier must be followed by an assignment operator: '='");
    }
    advance();

    Expr *expr = parseExpr(out);
    if (!expr) {
      return parsingError("Error parsing expression!");
    }
    out->attachAssignExpr(new AssignStmt(out, assignee, expr));
  }

  if (!match(keyword_output)) {
    return parsingError("Expected output statement!");
  }

  advance();
  Expr *outputExpr = parseExpr(out);
  out->attachOutputStmt(new OutputStmt(out, outputExpr));

  return true;
}

Expr *Parser::parseExpr(ProgramAST *out) {
  Expr *LHS;
  switch (currentToken->kind) {
    case INTEGER:
      LHS = new IntegerLiteralExpr(
          out, static_cast<int64_t>(std::stol(currentToken->tokenStr)));
      break;
    case ID:
      LHS = new AssigneeReferenceExpr(out, currentToken->tokenStr);
      break;
    default: {
      return nullptr;
    }
  }

  advance();

  if (match(NEWLINE)) {
    advance();
    return LHS;
  } else if (isOperator(currentToken->kind)) {
    return parseBinaryOpExpr(out, LHS, base);
  } else {
    delete LHS;
    return nullptr;
  }
}

Expr *Parser::parseBinaryOpExpr(ProgramAST *out, Expr *LHS, Precedence prec) {
  Precedence currPrec = getOperatorPrecedence(currentToken->kind);
  while (true) {
    if (currPrec < prec) return LHS;

    TokenKind op = currentToken->kind;
    advance();

    Expr *RHS;
    RHS = parseExpr(out);

    if (!RHS) return nullptr;
    Precedence prevPrec = currPrec;
    currPrec = getOperatorPrecedence(currentToken->kind);

    if (currPrec < prevPrec) {
      RHS = parseBinaryOpExpr(out, RHS, prevPrec);
      if (!RHS) return nullptr;
    }
    LHS = new BinaryOpExpr(out, LHS, op, RHS);
  }
}
}  // namespace edge
