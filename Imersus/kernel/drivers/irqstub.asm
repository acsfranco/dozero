section .text

extern keyboard_handle
extern clock_handle

%macro IRQ 2
global irq%1

irq%1:
  pusha
  call %2
  mov dx, 0x20
  mov al, 0x20
  out dx, al    ; Sinal EOI (End Of Interrupt) para o PIC master
  popa
  iret
%endmacro

IRQ 32, clock_handle
IRQ 33, keyboar_handle
