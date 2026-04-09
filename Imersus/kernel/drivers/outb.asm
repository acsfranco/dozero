global outb

section .text

outb:
  push ebp
  mov ebp, esp
  push ax
  push dx
  mov dx, [ebp + 8]   ; port
  mov al, [ebp + 12]  ; value
  out dx, al ;;;;;;;;;;;;;;;;;;;;; tinha colocado ax
  pop dx
  pop ax
  pop ebp
  ret
