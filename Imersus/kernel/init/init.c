#include "../proc/process.h"
#include "../tty/tty.h"
#include "../include/stddef.h"
#include "../drivers/vga.h"
#include "../include/unistd.h"

extern process_t current_process;
extern tty_t tty_default;

void kernel_init() {
  static process_t kernel_process;  
  static tty_t tty = {
    .driver_putchar = (void (*)(char, uint16_t, uint16_t, uint32_t, uint32_t))vga_putchar,
    .driver_setcursorpos = (void (*)(uint16_t, uint16_t))vga_setcursorpos,   //////// TINHA COLOCADO tty_setcursorpos
    .width = 80,
    .height = 25,
    .bkcolor = 0,
    .fgcolor = 7
  };
  
  tty_default = tty;

  static file_ops_t tty_ops_out = {
    .read = NULL,
    .write = (int (*)(void *, char *, int)) tty_write
  };

  static file_t fstdout = {
    .ctx = &tty_default,
    .ops = &tty_ops_out
  };
  
  static file_ops_t tty_ops_in = {
    .read = (int (*)(void *, char *, int)) tty_read,
    .write = NULL
  };

  static file_t fstdin = {
    .ctx = &tty_default,
    .ops = &tty_ops_in
  };

  kernel_process.fd_table[STDIN] = &fstdin;
  kernel_process.fd_table[STDOUT] = &fstdout;
  kernel_process.fd_table[STDERR] = &fstdout;

  current_process = kernel_process;
}
