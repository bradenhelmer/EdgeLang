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

#ifndef SYMBOL
#define SYMBOL(X, CHAR) TOKEN(X)
#endif

TOKEN(ID)
TOKEN(INTEGER)
TOKEN(NEWLINE)
TOKEN(UNKNOWN)
TOKEN(EDGEEOF)

OPERATOR(OPERATOR_START, '~')
OPERATOR(ASSIGN, '=')
OPERATOR(ADD, '+')
OPERATOR(SUB, '-')
OPERATOR(MUL, '*')
OPERATOR(DIV, '/')
OPERATOR(OPERATOR_END, '~')

SYMBOL(RPAREN, '(')
SYMBOL(LPAREN, ')')

KEYWORD(output)

#undef TOKEN
#undef OPERATOR
#undef KEYWORD
#undef SYMBOL
