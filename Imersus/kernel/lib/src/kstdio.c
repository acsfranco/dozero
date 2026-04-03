#include <stdarg.h>
#include "../../tty/tty.h"
#include "../include/kstdio.h"

extern void vga_putchar(char, int, int);

tty_t tty;

void print_char(char c) {
  tty_putchar(&tty, c);
}

void print_string(const char *str) {
  while (*str) {
    print_char(*str);
    str++;
  }
}

void print_int(int num) {
  char buffer[16];
  int i = 0;

  if (num == 0) {
    print_char('0');
    return;
  }

  if (num < 0) {
    print_char('-');
    num *= -1;
  }

  while (num > 0) {
    buffer[i++] = '0' + (num % 10);
    num /= 10;
  }

  while (i--) {
    print_char(buffer[i]);
  }
}

void print_hex(int num) {
  char buffer[16];
  char hex[] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

  int rest, quocient = num, i = 0;

  while (num > 15) {
    rest = num % 16;
    num /= 16;
    buffer[i++] = hex[rest];
  }

  buffer[i] = hex[num];
  while (i >= 0) {
    print_char(buffer[i--]);
  }
}

void print_float(double num, unsigned char nd) {
  if (num < 0) {
    print_char('-');
    num *= -1;
  }

  int inteiro = (int)num;
  print_int(inteiro);
  print_char('.');
  
  double frac = num - inteiro;
  for (int i = 0; i < nd; i++) {
    frac *= 10;
    int digit = (int)frac;
    print_char(digit + '0');
    frac -= digit;
  }
}

void kclear() {
  tty.driver_putchar = vga_putchar;
  tty.width = 80;
  tty.height = 25;

  tty_clear(&tty);
}

void kprintf(char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  unsigned char nd = 0;
  tty.driver_putchar = vga_putchar;
  tty.width = 80;
  tty.height = 25;

  while (*fmt) {
    if (*fmt == '%') {
      fmt++;
      if (*fmt == '.') {
        fmt++;
        while(*fmt >= '0' && *fmt <= '9') {
          nd = nd * 10 + (*fmt - '0');
          fmt++;
        }
      }
      if (*fmt == 'd') {
        int val = va_arg(args, int);
        print_int(val);
      }
      if (*fmt == 's') {
        char *val = va_arg(args, char*);
        print_string(val);
      }
      if (*fmt == 'f') {
        double val = va_arg(args, double);
        if (nd == 0) {
          nd = 6;
        }
        print_float(val, nd);
        nd = 0; ///////////////////////////// Corrige o problema de imprimir um número float com restrição de casas decimais, depois de imprimir um sem restrições
      }
      if (*fmt == 'c') {
        char val = (char)va_arg(args, int);
        print_char(val);
      }
      if (*fmt == 'x') {
        int val = va_arg(args, int);
        print_hex(val);
      }
      if (*fmt == '%') {
        print_char('%');
      }
    } else {
      print_char(*fmt);
    }
    fmt++;
  }
  va_end(args);
}
