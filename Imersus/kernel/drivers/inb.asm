global intb

section .text

outb:
  push ebp
  mov ebp, esp
  push dx
  mov dx, [ebp + 8]   ; port
  xor eax, eax
  in al, dx
  pop dx
  pop ebp
  ret
