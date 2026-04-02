CC = gcc
AS = nasm
LD = ld

CFLAGS = -m32 -ffreestanding
LDFLAGS = -m elf_i386

all: ./build/os.img

./build/boot.bin: ./boot/boot.asm
	$(AS) -f bin ./boot/boot.asm -o ./build/boot.bin

./build/kernel_entry.o: ./kernel/src/kernel_entry.asm
	$(AS) -f elf32 ./kernel/src/kernel_entry.asm -o ./build/kernel_entry.o

./build/vga_putchar.o: ./kernel/drivers/vga_putchar.asm
	$(AS) -f elf32 ./kernel/drivers/vga_putchar.asm -o ./build/vga_putchar.o

./build/kstdio.o: ./kernel/lib/src/kstdio.c
	$(CC) $(CFLAGS) -c ./kernel/lib/src/kstdio.c -o ./build/kstdio.o

./build/tty.o: ./kernel/tty/tty.c
	$(CC) $(CFLAGS) -c ./kernel/tty/tty.c -o ./build/tty.o

./build/kernel.o: ./kernel/src/kernel.c
	$(CC) $(CFLAGS) -c ./kernel/src/kernel.c -o ./build/kernel.o

./build/kernel.elf: ./build/kernel_entry.o ./build/kernel.o ./build/vga_putchar.o ./build/kstdio.o ./build/tty.o linker.ld
	$(LD) $(LDFLAGS) -T linker.ld ./build/kernel_entry.o ./build/kernel.o ./build/vga_putchar.o ./build/kstdio.o ./build/tty.o -o ./build/kernel.elf

./build/kernel.bin: ./build/kernel.elf
	objcopy -O binary ./build/kernel.elf ./build/kernel.bin

./build/os.img: ./build/boot.bin ./build/kernel.bin
	cat ./build/boot.bin ./build/kernel.bin > ./build/os.img

clean:
	rm ./build/*.o ./build/*.bin ./build/*.img
run:
	qemu-system-i386 -drive format=raw,file=./build/os.img
