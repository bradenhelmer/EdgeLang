// TokenDef.h
// ~~~~~~~~~~
// Token macro definitions.
#ifndef TOKEN

#define TOKEN(X)
#endif

#ifndef OPERATOR
#define OPERATOR(X, CHAR) TOKEN(X)
#endif

#ifndef KEYWORD
#define KEYWORD(X) TOKEN(keyword_##X)
#endif

TOKEN(ID)
TOKEN(INTEGER)
TOKEN(NEWLINE)
TOKEN(UNKNOWN)
TOKEN(EDGEEOF)

OPERATOR(ASSIGN, '=')
OPERATOR(ADD, '+')
OPERATOR(SUB, '-')
OPERATOR(MUL, '*')
OPERATOR(DIV, '/')

KEYWORD(output)

#undef TOKEN
#undef OPERATOR
#undef KEYWORD
