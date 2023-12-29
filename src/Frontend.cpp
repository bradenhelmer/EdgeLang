// Frontend.cpp
// ~~~~~~~~~~~~
// Implementations of frontend module
#include <Common.h>
#include <Frontend.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

  size_t idStrLen = out->getLength() + 1;
  char *idStr = new char[idStrLen];
  std::memcpy(idStr, out->start, out->getLength());

  if (MAP_FIND(keywords, idStr)) {
    out->kind = keywords.at(idStr);
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
  bufPtr = currPtr;
  return true;
}

void Lexer::lexAndPrintTokens() {
  Token tok;
  while (lexToken(&tok) && tok.kind != EDGEEOF) {
    std::cout.write(tok.start, tok.getLength());
    std::printf("\n");
  }
  bufPtr = sourceCode;
}
}  // namespace edge
