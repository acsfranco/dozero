#pragma once
#include <stdint.h>

void vga_putchar(char, uint16_t, uint16_t);
void vga_setcursorpos(uint16_t, uint16_t);
