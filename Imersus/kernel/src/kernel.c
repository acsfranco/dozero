#include "../lib/include/kstdio.h"

void kmain() {
  int x = 1234;
  double y = 1234.1234;
  const char *msg = "Mensagem";
  kclear();
  kprintf("Testando o kprintf\nVerificando os numeros nos seus diverso formatos\n%d\n%x\n%f\n%.2f\n%s\n%c\n",x,x,y,y,msg,'#');
  kprintf("\n\nAlexandre Franco");
  kclear();
  while(1);
}
