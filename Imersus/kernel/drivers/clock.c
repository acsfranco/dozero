#include "clock.h"
#include "../lib/include/kstdlib.h"

int x = 0; /////// APENAS PARA TESTE

void clock_handle() {
  kprintf("%d - ", x);
  x++;
}
