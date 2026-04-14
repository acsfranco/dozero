#include "tokens.h"
#include "../lib/include/kstdio.h"


int gettokens(char *cmdstr, char tokens[MAX_TOKENS][MAX_TKLEN]){
  int next = 0, i;
  while(*cmdstr) { /////////////// cmdstr no lugar de cmstr
    while (*cmdstr == ' ')
      cmdstr++;
    if (!*cmdstr)
      break;
    i = 0; ////// faltou essa linha de commando
    while (*cmdstr && *cmdstr != ' ' & i < MAX_TKLEN) {
      tokens[next][i++] = *cmdstr;
      cmdstr++; /////////////// faltou ;
    }
    tokens[next][i] = 0;
    next++;
  }

  return next;
}
