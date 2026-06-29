#include "pic.h"
#include "port.h"

void pic_config() {
  // Inicializando o estado ICW1 do PIC master e slave e fazendo o PIC esperar pelo
  // estado ICW2
  outb(0x20, 0x11); // ICW1 para o master
  outb(0xa0, 0x11); // ICW1 para o slave

  // Estado ICW2 - remapear os IRQs
  outb(0x21, 0x20);
  outb(0xa1, 0x28);

  // Estado ICW3 - configurando as conexões entre o pic master e o slave
  outb(0x21, 0x04); // diz ao pic master que tem um slave conectado a entrada IRQ2
  outb(0xa1, 0x02); // diz ao slave que seu irq está conectado a entrada IRQ2 do master

  // Estado ICW4 - configura o modo de operação
  outb(0x21, 0x01); // diz ao pic master que ele está trabalhando com o processador x98
  outb(0x21, 0x01); // diz ao pic slave que ele está trabalhando com o processador x86

  // Estado OCW1 - informar quais interrupções vão estar ativas
  outb(0x21, 0x00); // Todas as interrupções vão ser atendidas pelo PIC master
  outb(0xa1, 0x00); // Todas as interrupções vão ser atendidas pelo PIC slave

  // Estado OCW2 - avisar quando uma rotina de tratamento de interrupção terminar
  // final da rotina de tratamento da interrupção
  // outb(0x20, 0x20); // Sinal de EOI (End Of Interrupt)
}
