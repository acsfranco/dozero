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
  mov bh, 0x7         ;Cor de fundo e do caractere
  mov bl, [ebp + 8]  ;Caractere 
  mov esi, 0xb8000
  mov [esi + eax], bx
  popa
  pop ebp
  ret
