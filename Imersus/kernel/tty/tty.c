#include "tty.h"

tty_t tty_default;

void tty_putchar(tty_t *tty, char c) {
  if (c == '\n' || tty->x >= tty->width) {
    tty->y++;
    tty->x = 0;
  }
  if (c != '\n') {
    tty->driver_putchar(c, tty->x, tty->y);
    tty->x++;
  }
}

void tty_clear(tty_t *tty) {
  int x, y;
  for (y = 0; y < tty->height; y++) {
    for (x = 0; x < tty->width; x++) {
      tty_putchar(tty, ' ');
    }
  }
  tty->x = 0;
  tty->y = 0;
}

int tty_write(tty_t *tty, char *buf, int size) {
  for (int i = 0; i < size; i++) {
    tty_putchar(tty, buf[i]);
  }
}
