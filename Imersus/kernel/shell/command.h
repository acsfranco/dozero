#pragma once
#include <stdint.h>
#include "interpreter.h"

void exec_clear(uint8_t argc, char *argv[MAX_ARGS]);
void exec_version(uint8_t argc, char *argv[MAX_ARGS]);
void exec_setcolor(uint8_t argc, char *argv[MAX_ARGS]);
void exec_echo(uint8_t argc, char *argv[MAX_ARGS]);
