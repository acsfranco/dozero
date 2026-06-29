#include "keyboard.h"
#include "../tty/tty.h" //////////////////////////// tinha colocado kernel/tty/tty.h
#define KB_BUFFER_SIZE 256

#include "../lib/include/kstdio.h" ////// PARA TESTE

keyboard_t keyboard; /////////////////////////// Estava keyboart_t
void *kbctx;
char keyprior = 0;

char keybuff[KB_BUFFER_SIZE];
int head = 0;
int tail = 0;

void buffer_push(char c) {
  int next = (head + 1) % KB_BUFFER_SIZE;
  
  if (next != tail) {
    keybuff[head] = c;
    head = next;
  }
}

char buffer_pop() {
  if (head == tail) return 0;
  
  char c = keybuff[tail];
  tail = (tail + 1) % KB_BUFFER_SIZE;
  return c;
}

static const char keymap[128] = {
  0, 27, '1','2','3','4','5','6','7','8','9','0','-','=', '\b',
  '\t',
  'q','w','e','r','t','y','u','i','o','p','[',']','\n',
  0, // ctrl
  'a','s','d','f','g','h','j','k','l',';','\'','`',
  0, // shift
  '\\','z','x','c','v','b','n','m',',','.','/',
  0, // shift
  '*',
  0, // alt
  ' ',
  0, // caps
};

static const char keymap_shift[128] = {
  0, 27, '!','@','#','$','%','^','&','*','(',')','_','+','\b',
  '\t',
  'Q','W','E','R','T','Y','U','I','O','P','{','}','\n',
  0,
  'A','S','D','F','G','H','J','K','L',':','"','~',
  0,
  '|','Z','X','C','V','B','N','M','<','>','?',
  0,
  '*',
  0,
  ' ',
  0,
};

uint8_t get_scancode() {
  return inb(KEYBOARD_DATA_PORT);
}

char decode_scancode(uint8_t scancode) {
  if (scancode == 0x2a || scancode == 0x36)
    keyboard.shift = 1;
  else if (scancode == 0xaa || scancode == 0xB6)
    keyboard.shift = 0;
  else if (scancode & 0x80) { // Teclado liberado
    keyboard.released = 1;
    keyboard.pressed = 0;
    keyprior = 0;
    return 0;
  } else if (!(scancode & 0x80)) { // Teclado pressionado
    keyboard.released = 0;
    keyboard.pressed = 1;
  } else
    return 0;

  char c;

  if (keyboard.shift)
    c = keymap_shift[scancode];
  else
    c = keymap[scancode];
  
  if (c == keyprior && !keyboard.released)
    return 0;

  keyprior = c;

  keyboard.key = c;

  return c;
}

void keyboard_read() {
  uint8_t scancode = get_scancode();
  char c = decode_scancode(scancode);

  if (c)
      buffer_push(c);

  tty_push_char(kbctx, buffer_pop());
}

void keyboard_handle() {
  keyboard_read();
}

void keyboard_init(void *ctx) {
  kbctx = ctx;
  outb(KEYBOARD_STATUS_PORT, 0xae);
}
