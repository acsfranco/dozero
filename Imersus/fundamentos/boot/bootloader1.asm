ORG 0x7c00		; Avisa ao montador que o código deve começar em 0x7c00 (Endereço onde o
			; bootloader é carregado pela BIOS)
BITS 16			; Compila o código em 16 bits

; =================================================================================================
;	Código de um bootloader que mostra uma mensagem na tela,
;	espera o usuário pressionar uma tecla, entra no modo de
;	vídeo gráfico VGA 320x200x8 bits de cores (256 cores) e
;	preenche a tela com pixels coloridos.
;
;	Para compilar, digite no terminal: nasm -f bin bootloader1.asm -o boot.bin
;	Para executar no qemu, digite no terminal qemu-system-i386 -drive format=raw, file=boot.bin
; **************************************************************************************************

start:			; rótulo - É convertido pelo montador em uma posição de memória (neste caso
			; o endereço é 0x7c00, por causa dessa informação na diretiva ORG

; Zerando os registradores de segmento

	xor ax, ax
	mov ds, ax
	mov es, ax
	mov ss, ax

;	Escrevendo uma mensagem na tela

	mov ah, 0x0e	; Rotina de impressão de caractere na tela, na posição atual do cursor
	mov si, mens 
print:	mov al, [si]
	cmp al, 0
	jz keypress
	int 0x10	; Interrupção da bios de video
	inc si
	jmp print

; Aguardando uma tecla ser pressionada

keypress:
	mov ah, 0	; Rotina para esperar uma tecla ser pressionada
	int 0x16	; Interrupção da bios do teclado

; Mudando o modo de vídeo para o modo gráfico VGA 320x200x256 cores

modografico:
	mov ah, 0	; Rotina da bios para setar o modo de vídeo
	mov al, 0x13	; 320 x 200 x 8 bits de cores (256 cores)
	int 0x10
	mov ax, 0xa000	; 0xa0000 - Endereço da VRAM
	mov es, ax	; Atribuindo o valor 0xA000 no registrador es
	mov di, 0	; Atribuindo o valor inicial 0, de deslocamento, no registrador di. es * 16 + di = 0xA0000 

; Desenhando na tela

	mov cl, 1
loop:	mov byte es:[di], cl
	inc di
	inc cx
	cmp cx, 64000
	jz end
	jmp loop	
end:	jmp end
mens	db "Iniciando o sistema operacional Imersus!", 13, 10, "Pressione uma tecla para ir para o modo grafico...", 13, 10, 0 
times	510 - ($ - $$) db 0
	dw 0xAA55	; magic number de um código bootável
