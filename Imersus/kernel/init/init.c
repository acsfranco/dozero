#include "../proc/process.h"
#include "../tty/tty.h"
//#include "../drivers/vga.h" ///////////////////////////////////////////// Não existe - criar;

#define NULL ((void *)0)

extern void vga_putchar(char, int, int); /////////////////// Tirar depois

extern process_t current_process;
extern tty_t tty_default;


void kernel_init() {
  static process_t kernel_process;
  
  static tty_t tty = {
    .driver_putchar = vga_putchar,
    .width = 80,
    .height = 25
  }; ////////////////////////////// Faltou o ;
  
  tty_default = tty;

  static file_ops_t tty_ops_out = {
    .read = NULL,
    .write = (int (*)(void *, char *, int)) tty_write
  }; //////////////////////////// Faltou o ;

  static file_t fstdout = {
    .ctx = &tty_default,
    .ops = &tty_ops_out
  }; ////////////////// Faltou o ;

  kernel_process.fd_table[1] = &fstdout;
  kernel_process.fd_table[2] = &fstdout;

  current_process = kernel_process;
}
