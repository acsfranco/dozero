#pragma once

typedef struct tty {
  int x, y;
  int width, height;
  void (*driver_putchar)(char c, int x, int y);
} tty_t;

void tty_putchar(tty_t *tty, char c);
void tty_clear(tty_t *tty);
int tty_write(tty_t *tty, char *buf, int size);
