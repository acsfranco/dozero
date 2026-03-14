ORG 0x7c00		; Avisa ao montador que o código deve começar em 0x7c00 (Endereço onde o
			        ; bootloader é carregado pela BIOS)
BITS 16			  ; Compila o código em 16 bits

start:			  ; rótulo - É convertido pelo montador em uma posição de memória (neste caso
			        ; o endereço é 0x7c00, por causa dessa informação na diretiva ORG

; ==============================================================================================
; Esse código cria um bootloader psicodélico, com uma animação gráfica de cores e retângulos      
; Isso não fará parte do nosso sistema operacional, criamos esse código para aprender de uma      
; forma mais lúdica o que é convensão de chamada e consequentemente treinar criação de subrotina
; e uso de pilha em assembler.
; ==============================================================================================

; Zerando os registradores de segmento

	xor ax, ax
	mov ds, ax
	mov es, ax
	mov ss, ax
  mov sp, 0x7000

; Setando o modo de vídeo para gráfico VGA 320x200x256 cores

  mov ah, 0
  mov al, 0x13
  int 0x10

; chamando a função putpixel

  push 2    ; cor
  push 199  ; linha
  push 319  ; coluna
  call putpixel
  add sp, 6 ; limpando a pilha
  
; chamando a função rect

  push 1    ; cor azul
  push 150  ; yfim
  push 150  ; xfim
  push 100  ; y inicial
  push 100  ; x inicial
  call rect
  add sp, 10; limpando a pilha

; chamando novamente a função rect

  push 2
  push 180
  push 300
  push 80
  push 200
  call rect
  add sp, 10
  
; chamando o desenharect

  mov cx, 0
anima:
  push cx  ; cor inicial dos retângulos
  call desenharect;
  add sp, 2

  mov ah, 0 ; esperar uma tecla ser pressionada
  int 0x16

  inc cx
  jmp anima

  jmp end

; Função desenharetangulos - Desenhar um conjunto de retângulos empilhados e de tamanhos diferentes
;
; Parâmetro
;   corinicial - A cor do primeiro retângulos, os outros retângulos são valores incrementados da corinicial
;

desenharect:
  push bp
  mov bp, sp
  pusha             ; armazena os valores de todos os registradores na pilha
  mov ax, [bp + 4]  ; cor inicial
  mov bx, 0         ; x inicial 
  mov cx, 0         ; y inicial
  mov dx, 319       ; x final
  mov si, 199       ; y final
  xor di, di        ; contador
looprect:
  push ax           ; cor
  push si           ; y final
  push dx           ; x final
  push cx           ; y inicil
  push bx           ; x inicial
  call rect
  add sp, 10
  add bx, 5
  add cx, 5
  sub dx, 5
  sub si, 5
  inc ax
  inc di
  cmp di, 24
  jz fimdesenha
  jmp looprect
fimdesenha:
  popa
  pop bp
  ret

;
; Função putpixel - Desenha um pixel na tela
;
; Parâmetros
;   coluna - coluna onde o pixel vai ser desenhado
;   linha - linha onde o pixel vai ser desenhado
;   cor - cor do pixel (0 e 255)
;

putpixel:
  push bp
  mov bp, sp
  push ax
  push bx
  push dx
  push si
  push es
  mov bx, [vram]
  mov es, bx
  mov ax, [bp + 6] ; colocando o valor da linha em ax
  mov bx, 320
  mul bx
  add ax, [bp + 4] ; somando com o valor da coluna - o endereço da vram correspondente às coordenadas do pixel
  mov si, ax
  mov bl, [bp + 8] ; jogando a cor do pixel para bl
  mov es:[si], bl  ; jogando o valor de bl na vram (desenhar um pixel)
  pop es
  pop si
  pop dx
  pop bx
  pop ax
  pop bp
  ret

;
; rect - Desenha um retângulo na tela
;
; Parâmetros
;   x    - Coordenada x inicial
;   y    - Coordenada y inicial
;   xfim - Coordenada x final
;   yfim - Coordenada y final
;   cor  - Cor de preenchimento do retângulo
;

rect:
  push bp
  mov bp, sp
  push bx
  push cx
  push dx
  mov bx, [bp + 12] ; colocando a cor em bx
  mov dx, [bp + 6]  ; colocando o valor do y inicial em dx
linha:
  mov cx, [bp + 4]  ; colocando o valor do x inicial em cx
  cmp dx, [bp + 10] ; comparando a linha corrente com o y final
  jae saidesenha
coluna:
  push bx           ; cor do pixel
  push dx           ; y do pixel
  push cx           ; x do pixel
  call putpixel
  add sp, 6
  inc cx
  cmp cx, [bp + 8]
  jz  inclinha
  jmp coluna
inclinha:
  inc dx
  jmp linha
saidesenha:
  pop dx
  pop cx
  pop bx
  pop bp
  ret

end:	jmp end
vram dw 0xa000
times	510 - ($ - $$) db 0
dw 0xAA55	; magic number de um código bootável
