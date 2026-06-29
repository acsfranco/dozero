#include "isr.h"
#include "../lib/include/kstdio.h"

void isr_handle(uint32_t intnum, uint32_t errcode) {
  kprintf("\nHOUVE UMA EXCECAO:\n");
  switch(intnum) {
    case 0:
      kprintf("Causa: Divisao por zero\n");
      break;
    case 13:
      kprintf("Causa: Falha Geral de Protecao\n");
      break;
    case 14:
      kprintf("Causa: Falta de pagina\n");
      break;
    default:
      kprintf("Causa: Desconhecida\n");
  }
  kprintf("Excecao No.: %d\n", intnum);
  kprintf("Codigo de erro: %d\n", errcode);
  while(1);
}
