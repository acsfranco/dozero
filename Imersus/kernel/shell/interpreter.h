#pragma once
#include <stdint.h>
#include "tokens.h"
#define MAX_ARGS 17

typedef struct {
  char *identifier;
  uint8_t argc;
  char *argv[MAX_ARGS];
} command_t;

typedef struct {
  char *identifier;
  void (*executor)(uint8_t argc, char *argv[MAX_ARGS]);
} exec_t;

void decode(char tokens[MAX_TOKENS][MAX_TKLEN], command_t *command, int noftokens);
int execute(command_t);
int load(command_t);
int run(char *strcmd);
