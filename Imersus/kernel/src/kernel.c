#include "../lib/include/kstdio.h"
#include "../init/init.h"
#include "../tty/tty.h"
#include "../shell/kshell.h"

extern tty_t tty_default;

void kmain() {
  kernel_init();
  kclear();
  kshell();
  while(1);
}
