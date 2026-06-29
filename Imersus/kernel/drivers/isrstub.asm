section .text

extern isr_handle

%macro ISR_NOCODE 1
global isr%1

isr%1:
  push 0  ; Fake error code
  push %1 ; Interrupt number
  jmp isr_common_stub
%endmacro

%macro ISR_CODE 1
global isr%1

isr%1:
  push %1 ; Interrupt number
  jmp isr_common_stub
%endmacro

isr_common_stub:
  pusha
  push dword [esp + 36]
  push dword [esp + 36]
  call isr_handle
  add esp, 8
  popa
  add esp, 8
  iret

ISR_NOCODE 0            ; Divisão por zero
ISR_NOCODE 1            ; Debug
ISR_NOCODE 2            ; NMI
ISR_NOCODE 3            ; Breakpoint
ISR_NOCODE 4            ;
ISR_NOCODE 5            ;
ISR_NOCODE 6            ;
ISR_NOCODE 7            ;
ISR_CODE   8            ;
ISR_NOCODE 9            ;
ISR_CODE   10           ;
ISR_CODE   11           ;
ISR_CODE   12           ;
ISR_CODE   13           ;
ISR_CODE   14           ;
ISR_NOCODE 15           ;
ISR_NOCODE 16           ;
ISR_CODE   17           ;
ISR_NOCODE 18           ;
ISR_NOCODE 19           ;
ISR_NOCODE 20           ;
ISR_CODE   21           ;
ISR_NOCODE 22           ;
ISR_NOCODE 23           ;
ISR_NOCODE 24           ;
ISR_NOCODE 25           ;
ISR_NOCODE 26           ;
ISR_NOCODE 27           ;
ISR_NOCODE 28           ;
ISR_NOCODE 29           ;
ISR_NOCODE 30           ;
ISR_NOCODE 31           ;
