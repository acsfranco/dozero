#include "../lib/include/kstdio.h"
#include "../init/init.h"
#include "../tty/tty.h"

extern tty_t tty_default;

void kmain() {
  int x = 1234;
  double y = 1234.1234;
  const char *msg = "Mensagem";
  kernel_init();
  kclear();
  tty_textcolor(&tty_default, 3);
  tty_backgroundcolor(&tty_default, 11);
  kprintf("Testando o kprintf\nVerificando os numeros nos seus diverso formatos\n%d\n%x\n%.2f\n%f\n%s\n%c\n",x,x,y,y,msg,'#');
  kprintf("\n\nAlexandre Franco");
  while(1);
}
