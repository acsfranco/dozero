#include "tty.h"

tty_t tty_default;

void tty_setcursorpos(tty_t *tty, uint16_t x, uint16_t y) {
  tty->driver_setcursorpos(x, y);
}

void tty_push_char(tty_t *tty, char c) {
  int next = (tty->buff.head + 1) % TTY_BUFFER_SIZE;
  if (next != tty->buff.tail) {
    tty->buff.buffer[tty->buff.head] = c;
    tty->buff.head = next;
  }
}

int tty_read(tty_t *tty, char *buf, int size) {
  int i = 0;
  while (i < size) {
    while (tty->buff.head == tty->buff.tail); // Bloquei enquanto o buffer estiver vazio
    buf[i++] = tty->buff.buffer[tty->buff.tail];
    tty->buff.tail = (tty->buff.tail + 1) % TTY_BUFFER_SIZE;

    if (buf[i - 1] == '\n')
      break;
  }
  return i;
}

void tty_putchar(tty_t *tty, char c) {
  if (c == '\n' || tty->x >= tty->width) {
    tty->y++;
    tty->x = 0;
  }
  if (c != '\n') {
    tty->driver_putchar(c, tty->x, tty->y, tty->bkcolor, tty->fgcolor);
    tty->x++;
  }
  tty_setcursorpos(tty, tty->x, tty->y);
}

void tty_textcolor(tty_t *tty, uint32_t color){
  tty->fgcolor = color;
}

void tty_backgroundcolor(tty_t *tty, uint32_t color){
  tty->bkcolor = color;
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
  tty_setcursorpos(tty, tty->x, tty->y);
}

int tty_write(tty_t *tty, char *buf, int size) {
  for (int i = 0; i < size; i++) {
    tty_putchar(tty, buf[i]);
  }
}
