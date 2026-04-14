#include "../tty/tty.h"
#include "../lib/include/kstdio.h"
#include "../global/global.h"
#include "interpreter.h"
#include <stdint.h>

extern tty_t tty_default;

void header() {
  // Piramide
  tty_textcolor(&tty_default, 3);
  tty_backgroundcolor(&tty_default, 15);
  kprintf("                        Shell: KShell - versao %.1f                              \n", KSHELL_VERSION);
  tty_textcolor(&tty_default, 11);
  tty_backgroundcolor(&tty_default, 1);
  kprintf("                   %c%c%c%c%c%c%c%c%c%c                                                   \n", 220, 219, 219, 219, 219, 219, 219, 219, 219, 220);
  kprintf("                  %c%c%c%c%c%c%c%c%c%c%c%c         Autor: %s                  \n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, AUTHOR);
  kprintf("                 %c%c%c%c%c%c%c%c%c%c%c%c%c%c        SO versao %.1f                            \n", 219, 219, 219, 219, 219, 219, 178, 178, 178, 178, 178, 178, 219, 219, SO_VERSION);
  kprintf("                %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c       Kernel versao %.1f                        \n", 219, 219, 219, 219, 219, 219, 178, 178, 178, 178, 178, 178, 178, 178, 219, 219, KERNEL_VERSION);
  kprintf("               %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c      Data: %s                         \n", 219, 219, 219, 219, 219, 219, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 219, 219, DATE);
  kprintf("              %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c                                              \n", 219, 219, 219, 219, 219, 219, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 177, 219, 219);
  kprintf("             %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c    Fonte: %s                           \n", 219, 219, 219, 219, 219, 219, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 219, 219, SOURCE);
  kprintf("            %c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c                                            \n", 219, 219, 219, 219, 219, 219, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 176, 219, 219);
  kprintf("                                                                                ");
  // Nome IMERSUS
  kprintf("  %c%c %c%c%c    %c%c%c  %c%c%c%c%c%c%c  %c%c%c%c%c%c   %c%c%c%c%c%c  %c%c    %c%c  %c%c%c%c%c%c                     \n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219);
  kprintf("  %c%c %c%c%c%c  %c%c%c%c  %c%c       %c%c   %c%c %c%c       %c%c    %c%c %c%c                          \n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219);
  kprintf("  %c%c %c%c %c%c%c%c %c%c  %c%c%c%c%c%c%c  %c%c%c%c%c%c   %c%c%c%c%c%c  %c%c    %c%c  %c%c%c%c%c%c%c                    \n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219);
  kprintf("  %c%c %c%c  %c%c  %c%c  %c%c       %c%c   %c%c       %c%c %c%c    %c%c       %c%c                    \n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219);
  kprintf("  %c%c %c%c      %c%c  %c%c%c%c%c%c%c  %c%c   %c%c  %c%c%c%c%c%c   %c%c%c%c%c%c   %c%c%c%c%c%c%c                    \n\n", 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219);
  tty_backgroundcolor(&tty_default, 0);
}

void prompt() {
  uint32_t color = tty_default.fgcolor;
  tty_textcolor(&tty_default, 2);
  kprintf("root@dozero");
  tty_textcolor(&tty_default, 7);
  kprintf(":");
  tty_textcolor(&tty_default, 3);
  kprintf("~/");
  tty_textcolor(&tty_default, 7);
  kprintf("$ ");
  tty_default.fgcolor = color;
}

void kshell() {
  header();
  prompt();
  run("echo Seja bem vindo ao kshell");
  prompt();
  run("version");
  run("clear");
  prompt();
  run("setcolor 0 4");
  run("version");
}
