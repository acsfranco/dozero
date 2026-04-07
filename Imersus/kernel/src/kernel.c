#include "../lib/include/kstdio.h"
#include "../init/init.h"

void kmain() {
  int x = 1234;
  double y = 1234.1234;
  const char *msg = "Mensagem";
  kernel_init();
  kclear();
  kprintf("Testando o kprintf\nVerificando os numeros nos seus diverso formatos\n%d\n%x\n%.2f\n%f\n%s\n%c\n",x,x,y,y,msg,'#');
  kprintf("\n\nAlexandre Franco");
  kclear();
  while(1);
}
