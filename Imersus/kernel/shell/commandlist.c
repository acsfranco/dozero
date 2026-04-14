#include "interpreter.h"
#include "../include/stddef.h"
#include "command.h" /////////////// Faltou esse include

exec_t commandlist[] = { /////////////////////// exec_t e não exect_t
  {
    .identifier = "clear",
    .executor = exec_clear
  },
  {
    .identifier = "version",
    .executor = exec_version
  },
  {
    .identifier = "setcolor",
    .executor = exec_setcolor
  },
  {
    .identifier = "echo",
    .executor = exec_echo
  },
  {
    .identifier = "",
    .executor = NULL /////////////////////// executor e não execute
  }
};
