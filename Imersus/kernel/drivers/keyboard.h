#include "port.h"

#define KEYBOARD_DATA_PORT 0x60
#define KEYBOARD_STATUS_PORT 0x64

typedef struct {
  int scancode;
  char key;
  int shift;
  int pressed;
  int released;
} keyboad_t;

uint8_t get_scandcode();
char decode_scancode(uint8_t scancode);

void keyboard_handle(); // Para implementação quando trabalhar com interrupção
void keyboard_read();
void keyboard_init();
