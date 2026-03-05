void print(const char *msg);
void init_videomode(unsigned char mode);
void wait_key();
void init_lines();
void putpixel(unsigned short x, unsigned short y, unsigned char color);
void wait_retrace();
void paint();
void anime();
unsigned short lines[200];
unsigned char *vram = (unsigned char *)0xA0000;

void kmain() {
   print("Iniciando o sistema operacional Imersus...\n");
   print("Pressione alguma tecla para iniciar modo grafico...\n");
   wait_key();
   init_videomode(0x13);
   init_lines();
   paint();
   anime();
   while(1);
}

void print(const char *msg) {
  while(*msg != 0) {
    __asm__ __volatile__(
      "movb $0x0E, %%ah \n"
      "movb %0, %%al    \n"
      "int $0x10        \n"
      "cmp $'\n', %%al  \n"
      "jnz sai          \n"
      "mov $0xd, %%al   \n"
      "int $0x10        \n"
      "sai:             \n"
      :
      :"b"(*msg)
      :"ax", "memory"
    );
    msg++;
  }
}

void init_videomode(unsigned char mode) {
  __asm__ __volatile__(
    "movb $0, %%ah \n"
    "movb %0, %%al \n"
    "int $0x10     \n"
    :
    :"b"(mode)
    :"ax"
  );
}

void wait_retrace() {
  __asm__ __volatile__(
    "mov $0x3DA, %%dx  \n"
    ".wait1:           \n"
    "in %%dx, %%al     \n"
    "test $0x08, %%al  \n"
    "jnz .wait1        \n"
    ".wait2:           \n"
    "in %%dx, %%al     \n"
    "test $0x08, %%al  \n"
    "jz .wait2         \n"
    :
    :
    :"ax", "dx"
  );
}

void wait_key() {
  __asm__ __volatile__(
    "movb $0, %%ah \n"
    "int $0x16     \n"
    :
    :
    :"ax"
  );
}

void init_lines() { 
  for (unsigned short i = 0; i < 200; i++) {
    lines[i] = i * 320;
  }
}

void putpixel(unsigned short x, unsigned short y, unsigned char color) {
  vram[lines[y] + x] = color;
}

void paint() {
  for (unsigned short x = 0; x < 320; x++) {
    for (unsigned short y = 0; y < 200; y++) {
      unsigned char color = x / 32 + 1;
      putpixel(x, y, color);
    }
  }
}

void anime() {
  unsigned char color;

  for (unsigned short i = 0; i < 640; i++) {
    for (unsigned short x = 0; x < 319; x++) {
      for (unsigned short y = 0; y < 200; y++) {
        color = vram[lines[y] + 0];
        vram[lines[y] + x] = vram[lines[y] + x + 1];
        vram[lines[y] + 319] = color;
      }
    }
    //wait_retrace();
    //render();
  }
}
