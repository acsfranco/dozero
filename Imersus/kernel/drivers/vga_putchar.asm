BITS 32

section .text
GLOBAL vga_putchar

; Função que implementa o driver de comunicação com o controlador de vídeo em modo texto
; para imprimir um caractere na tela
;
; Parâmetro:
;   caracter - caracter a ser impresso
;   x - posição x do cursor
;   y - posição y do cursor
;   bkc - cor de fundo
;   fgc - cor do caractere

vga_putchar:
  push ebp
  mov ebp, esp
  pusha
  xor eax, eax
  mov cx, [ebp + 12]  ;Posição x do cursor
  mov ax, [ebp + 16]  ;Posição y do cursor
  mov bx, 160
  mul bx              ;ax = y * 160
  shl cx, 1           ;multiplicar cx por 2
  add ax, cx          ;ax = y * 160 + (x * 2)
  mov bh, [ebp + 20]  ;Cor de fundo
  shl bh, 4
  or bh, [ebp + 24]
  mov bl, [ebp + 8]   ;Caractere 
  mov esi, 0xb8000
  mov [esi + eax], bx
  popa
  pop ebp
  ret
