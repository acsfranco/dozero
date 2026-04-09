#pragma once
#include <stdint.h>
#define TTY_BUFFER_SIZE 1024

typedef struct {
  char buffer[TTY_BUFFER_SIZE];
  int head, tail;
} buff_quee;

typedef struct tty {
  uint16_t x, y;
  uint16_t width, height;
  uint32_t bkcolor, fgcolor;
  
  buff_quee buff;
  void (*driver_putchar)(char, uint16_t, uint16_t, uint32_t, uint32_t);
  void (*driver_setcursorpos)(uint16_t, uint16_t);
} tty_t;

void tty_push_char(tty_t *, char);
void tty_putchar(tty_t *, char);
void tty_textcolor(tty_t *, uint32_t);
void tty_backgroundcolor(tty_t *, uint32_t);
void tty_clear(tty_t *);
void tty_setcursorpos(tty_t *, uint16_t, uint16_t); ////// ESQUECI ;
int tty_write(tty_t *, char *, int);
int tty_read(tty_t *, char *, int);

