CC = gcc
AS = nasm
LD = ld

CFLAGS = -m16 -ffreestanding -fno-builtin -nostdlib -fno-pic
LDFLAGS = -m elf_i386

all: ../bin/os.img

../bin/boot.bin: boot/boot.asm
	$(AS) -f bin boot/boot.asm -o ../bin/boot.bin

../bin/kernel.o: kernel/init/kernel.c
	$(CC) $(CFLAGS) -c kernel/init/kernel.c -o ../bin/kernel.o

../bin/kernel.elf: ../bin/kernel.o linker.ld
	$(LD) $(LDFLAGS) -T linker.ld ../bin/kernel.o -o ../bin/kernel.elf

../bin/kernel.bin: ../bin/kernel.elf
	objcopy -O binary ../bin/kernel.elf ../bin/kernel.bin

../bin/os.img: ../bin/boot.bin ../bin/kernel.bin
	cat ../bin/boot.bin ../bin/kernel.bin > ../bin/os.img

clean:
	rm ../bin/*.*
run:
	qemu-system-i386 -drive format=raw,file=../bin/os.img
