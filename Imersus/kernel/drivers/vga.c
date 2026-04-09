#include "port.h"

void vga_setcursorpos(uint16_t x, uint16_t y) {
  uint16_t pos = y * 80 + x;
  outb(0x3d4, 0x0f); // Controladora está preparada para receber os 8 bits menos significativos de pos ////////// ESQUECI ; 
  outb(0x3d5, (uint8_t)(pos & 0xff)); //////////// ESQUECI )
  outb(0x3d4, 0x0e); // Controladora está preparada para receber os 8 bits mais significativos de pos
  outb(0x3d5, (uint8_t)((pos >> 8) & 0xff)); ////////// ESQUECI DO CAST
}
