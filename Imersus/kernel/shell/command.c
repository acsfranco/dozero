#include "command.h"
#include <stdint.h>
#include "../tty/tty.h"
#include "../lib/include/kstdio.h"
#include "../global/global.h"

extern tty_t tty_default;

void exec_clear(uint8_t argc, char *argv[MAX_ARGS]) {
  kclear();
}

void exec_version(uint8_t argc, char *argv[MAX_ARGS]) {
  kprintf("\nImersus versao %.1f\n", SO_VERSION);
  kprintf("Kernel versao %.1f\n", KERNEL_VERSION);
  kprintf("KShell versao %.1f\n", KSHELL_VERSION);
  kprintf("Autor %s\n", AUTHOR);
  kprintf("Fonte %s\n", SOURCE);
  kprintf("Data de criacao %s\n", DATE);
}

void exec_setcolor(uint8_t argc, char *argv[MAX_ARGS]){
  const char *strfgcolor = argv[0];
  const char *strbkcolor = argv[1];

  uint32_t fgcolor = 0, bkcolor = 0;

  while(*strfgcolor) {
    fgcolor = fgcolor * 10 + (*strfgcolor - '0');
    strfgcolor++;
  }

  while(*strbkcolor) {
    bkcolor = bkcolor * 10 + (*strbkcolor - '0');
    strbkcolor++;
  }

  tty_textcolor(&tty_default, fgcolor);
  tty_backgroundcolor(&tty_default, bkcolor);
}


void exec_echo(uint8_t argc, char *argv[MAX_ARGS]) {
  kprintf("\n");
  for(int i = 0; i < argc; i++) {
    kprintf("%s ", argv[i]);
  }
  kprintf("\n");
}
