#include "tty.h"

void tty_putchar(tty_t *tty, char c) { ////////////////////////////// Passagem de parâmetro por referência
  if (c == '\n' || tty->x >= tty->width) {
    tty->y++;
    tty->x = 0;
  }
  if (c != '\n') {
    tty->driver_putchar(c, tty->x, tty->y);
    tty->x++;
  }
}

void tty_clear(tty_t *tty) { //////////////////// Passagem de parâmetro por referência
  int x, y;
  for (y = 0; y < tty->height; y++) {
    for (x = 0; x < tty->width; x++) {
      tty_putchar(tty, ' ');
    }
  }
  tty->x = 0;
  tty->y = 0;
}
