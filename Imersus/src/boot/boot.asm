ORG	0x7C00
BITS	16			; Código em 16 bits
start:
	xor ax, ax
	mov ds, ax		; Zerando o registrador de segmento de dados
	mov ss, ax		; Zerando o registrador de segmento de pilha
	mov es, ax		; Zerando o registrador de segmento de dados extra
	mov sp, 0x7A00		; Ponteiro da pilha iniciando em 0x7C00

  mov dl, 0x80          ; Se for usar o pendrive para rodar numa máquina real, senão comente essa linha
  mov [BOOT_DRIVER], dl
	; Lendo o inicializador do kernel para a memória
	mov ah, 0x02		; Função da BIOS para leitura física no dispositivo de armazenamento
	mov al, 4		; Número de setores a serem lidos
	mov ch, 0		; Número do cilíndro
	mov cl, 2		; Setor inicial a ser lido
	mov dh,	0		; Cabeça de leitura 0
	mov dl, [BOOT_DRIVER]	; Driver - 0x80 - PenDrive
	mov bx, 0x1000		; Endereço de offset de memória para onde vai ser carregado o inicializado do kernel
	; O registrador de segmento es é o segmento de memória onde o kernel vai ser carregado - Seu valor já foi inicializado como 0
	int 0x13		; Executando a rotina da bios para gravar o kernel na memória
	
  mov ah, 0x0E
  mov al, 'O'
  int 0x10;
  mov al, 'I'
  int 0x10
  mov al, '!'
  int 0x10
  mov al, 10
  int 0x10
  mov al, 13
  int 0x10

  call 0x0000:0x1000		; Dando um salto para o kernel ser executado

BOOT_DRIVER: db 0
times	510-($-$$) db 0
dw 0xAA55			; Magic bootloader number

	
