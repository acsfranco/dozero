ORG 0x7c00		      ; Avisa ao montador que o código deve começar em 0x7c00 (Endereço onde o
			              ; bootloader é carregado pela BIOS)
BITS 16			        ; Compila o código em 16 bits

start:			        ; rótulo - É convertido pelo montador em uma posição de memória (neste caso
			              ; o endereço é 0x7c00, por causa dessa informação na diretiva ORG

; Zerando os registradores de segmento

	xor ax, ax
	mov ds, ax
	mov es, ax
	mov ss, ax
  mov sp, 0x7000

; Carregando o inicializador do Kernel na memória
 
  mov ah, 2
  mov al, 4
  mov cl, 2
  mov ch, 0
  mov dh, 0
  mov dl, 0x80
  mov bx, 0x1000
  int 0x13

  jmp 0x0000:0x1000 ; Rodar o nosso kernel
end:	jmp end
times	510 - ($ - $$) db 0
dw 0xAA55	; magic number de um código bootável
