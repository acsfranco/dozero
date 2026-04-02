BITS 16

section .text
GLOBAL _start   ;exporta o _start
EXTERN kmain    ;importa o kmain

_start:
  mov ah, 0x0e
  mov si, msg
nextchar:
  mov al, [si]
  cmp al, 0
  jz loadgdt
  inc si
  int 0x10
  jmp nextchar

loadgdt:
;Carregar o GDT
  cli
  
  lgdt [gdt_descritor]

;Configurar o registrador CR0 para setar o modo protegido
  
  mov eax, cr0
  or eax, 0x1
  mov cr0, eax	;Setamos o bit menos significativo de CR0 para 1 - Modo protegido

;JMP para a primeira região de memória a ser executada em modo protegido 

;Primeira região a ser executada em modo protegido
  jmp 0x08:gotokernel

BITS 32
gotokernel:
  mov ax, 0x10
  mov ds, ax
  mov es, ax
  mov gs, ax
  mov fs, ax
  mov ss, ax

  mov esp, 0x90000

  call kmain
  jmp $

section .data

gdt_start:	;Começo da tabela de descritores globais
	dq 0x0000000000000000	;Descritor nulo
	dq 0x00CF9A000000FFFF	;Descritor de código
	dq 0x00CF92000000FFFF	;Descritor de dados
gdt_end:

gdt_descritor:
	dw gdt_end - gdt_start - 1
	dd gdt_start
msg db "Carregando o kernel do Imersus, versao 1.0...", 10, 13, 0
