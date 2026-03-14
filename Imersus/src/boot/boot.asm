ORG 0x7c00    ; Endereço inicial do bootloader - Sem isso, o label start teria o endereço 0
BITS 16       ; Código em 16 bits

start:        ; Aqui começa o código do bootloader. Se alguma instrução saltar para start, vai saltar para 0x7c00 
  ; Zerando os registradores de segmento
  
  xor ax, ax
  mov es, ax
  mov ds, ax
  mov ss, ax
  mov sp, 0x7000    ; Endereço da pilha do bootloader

  ; Imprimindo uma mensagem
  
  mov ah, 0x0e
  mov si, mensagem  ; Endereço inicial da mensagem
print:
  mov al, [si]      ; O caractere corrente da mensagem a ser impresso
  cmp al, 0         ; Comparando o caractere corrente com 0 (final da mensagem)
  jz  carregakernel ; Se for final da mensagem salte para o carregamento do kernel
  int 0x10          ; Imprimindo o caractere corrente
  inc si            ; Incrementando si para ir para o endereço do próximo caractere da mensagem
  jmp print         ; Volta para imprimir o próximo caractere da mensagem

  ; Carregando o kernel

carregakernel:
  mov ah, 0         ; Espera uma tecla ser pressionada
  int 0x16

  mov ah, 2         ; Rotina na BIOS para carregar dados a partir de um setor físico do disco
  mov al, 4         ; Quantidade de setores que ocupa o kernel
  mov ch, 0         ; Cilindro
  mov cl, 2         ; Setor inicial do disco onde começa o kernel
  mov dh, 0         ; Cabeça
  mov dl, 0x80      ; 0x80 - Geralmente é o primeiro driver inicializável encontrado
  xor bx, bx        ; Endereço de segmento na memória para onde será carregado o kernel
  mov es, bx
  mov bx, 0x1000    ; Endereço de offset para onde será carregado o kernel
  int 0x13          ; Carregando o kernel

  mov sp, 0x1FFF    ; Endereço da pilha do kernel
  jmp 0x1000        ; Saltando para o kernel

  jmp $
mensagem db "Iniciando o sistema operacional Imersus...", 10, 13, "Pressione uma tecla para continuar...", 0
times 510 - ($ - $$) db 0 ; Completando com 0's até o tamanho do código chegar a 512 bytes
dw 0xaa55                 ; Bootloader magic number
