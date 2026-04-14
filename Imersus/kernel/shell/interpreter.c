#include "interpreter.h"
#include "../lib/include/kstring.h"
#include "../include/stddef.h" /////////////// Faltou esse include
#include "../lib/include/kstdio.h"

extern exec_t commandlist[]; //////// exec_t no lugar de exect_t e trocar o L por l em commandlist

void decode(char tokens[MAX_TOKENS][MAX_TKLEN], command_t *command, int noftokens) {
  for (int i = 0; i < noftokens; i++) { ///// colocar o int ante do i
    if (i == 0)
      command->identifier = tokens[i]; ///////// identifier no lugar de identifer
    else
      command->argv[i - 1] = tokens[i];
  }
  command->argc = noftokens - 1;
}

int execute(command_t command) {
  exec_t exec;
  int found = 0, i = 0;
  while (!found && commandlist[i].executor != NULL) { /////// executor no lugar de execute
    found = strequals(commandlist[i].identifier, command.identifier); ///////////////////// coloquei um = no lugar da ,
    i++;
  }

  if (found){
    commandlist[i - 1].executor(command.argc, command.argv);
    return 1;
  }
  return load(command);
}

int load(command_t command) {
  return 0;
}

int run(char *cmdstr) {
  char tokens[MAX_TOKENS][MAX_TKLEN];
  command_t command;
  int noftokens = gettokens(cmdstr, tokens);
  decode(tokens, &command, noftokens);
  if (!execute(command))
    return 0;
  return 1;
}
