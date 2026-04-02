#pragma once

typedef struct tty {
  int x, y;
  int width, height;
  void (*driver_putchar)(char c, int x, int y); //////// esqueci o void
} tty_t;

void tty_putchar(tty_t *tty, char c); /////////// passagem de parâmetro por referência
void tty_clear(tty_t *tty); //////////////////// passagem de parâmetro por referência
