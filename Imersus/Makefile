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

./build/irqstub.o: ./kernel/drivers/irqstub.asm
	$(AS) -f elf32 ./kernel/drivers/irqstub.asm -o ./build/irqstub.o

./build/isrstub.o: ./kernel/drivers/isrstub.asm
	$(AS) -f elf32 ./kernel/drivers/isrstub.asm -o ./build/isrstub.o

./build/kstdio.o: ./kernel/lib/src/kstdio.c
	$(CC) $(CFLAGS) -c ./kernel/lib/src/kstdio.c -o ./build/kstdio.o

./build/idt.o: ./kernel/drivers/idt.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/idt.c -o ./build/idt.o

./build/isr.o: ./kernel/drivers/isr.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/isr.c -o ./build/isr.o

./build/pic.o: ./kernel/drivers/pic.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/pic.c -o ./build/pic.o

./build/clock.o: ./kernel/drivers/clock.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/clock.c -o ./build/clock.o

./build/tty.o: ./kernel/tty/tty.c
	$(CC) $(CFLAGS) -c ./kernel/tty/tty.c -o ./build/tty.o

./build/init.o: ./kernel/init/init.c
	$(CC) $(CFLAGS) -c ./kernel/init/init.c -o ./build/init.o

./build/fs.o: ./kernel/fs/fs.c
	$(CC) $(CFLAGS) -c ./kernel/fs/fs.c -o ./build/fs.o

./build/process.o: ./kernel/proc/process.c
	$(CC) $(CFLAGS) -c ./kernel/proc/process.c -o ./build/process.o

./build/commandlist.o: ./kernel/shell/commandlist.c
	$(CC) $(CFLAGS) -c ./kernel/shell/commandlist.c -o ./build/commandlist.o

./build/command.o: ./kernel/shell/command.c
	$(CC) $(CFLAGS) -c ./kernel/shell/command.c -o ./build/command.o

./build/interpreter.o: ./kernel/shell/interpreter.c
	$(CC) $(CFLAGS) -c ./kernel/shell/interpreter.c -o ./build/interpreter.o

./build/kshell.o: ./kernel/shell/kshell.c
	$(CC) $(CFLAGS) -c ./kernel/shell/kshell.c -o ./build/kshell.o

./build/tokens.o: ./kernel/shell/tokens.c
	$(CC) $(CFLAGS) -c ./kernel/shell/tokens.c -o ./build/tokens.o

./build/kstring.o: ./kernel/lib/src/kstring.c
	$(CC) $(CFLAGS) -c ./kernel/lib/src/kstring.c -o ./build/kstring.o

./build/inb.o: ./kernel/drivers/inb.asm
	$(AS) -f elf32 ./kernel/drivers/inb.asm -o ./build/inb.o

./build/outb.o: ./kernel/drivers/outb.asm
	$(AS) -f elf32 ./kernel/drivers/outb.asm -o ./build/outb.o

./build/vga.o: ./kernel/drivers/vga.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/vga.c -o ./build/vga.o

./build/kernel.o: ./kernel/src/kernel.c
	$(CC) $(CFLAGS) -c ./kernel/src/kernel.c -o ./build/kernel.o

./build/keyboard.o: ./kernel/drivers/keyboard.c
	$(CC) $(CFLAGS) -c ./kernel/drivers/keyboard.c -o ./build/keyboard.o

./build/kernel.elf: ./build/kernel_entry.o ./build/kernel.o ./build/vga_putchar.o ./build/kstdio.o ./build/tty.o ./build/init.o ./build/fs.o ./build/inb.o ./build/outb.o ./build/vga.o ./build/kshell.o ./build/process.o ./build/commandlist.o ./build/command.o ./build/interpreter.o ./build/tokens.o ./build/kstring.o ./build/keyboard.o ./build/idt.o ./build/pic.o ./build/isr.o ./build/irqstub.o ./build/isrstub.o ./build/clock.o  linker.ld
	$(LD) $(LDFLAGS) -T linker.ld ./build/kernel_entry.o ./build/kernel.o ./build/vga_putchar.o ./build/kstdio.o ./build/tty.o ./build/init.o ./build/fs.o ./build/inb.o ./build/outb.o ./build/vga.o ./build/kshell.o ./build/process.o ./build/commandlist.o ./build/command.o ./build/interpreter.o ./build/tokens.o ./build/kstring.o ./build/keyboard.o ./build/idt.o ./build/pic.o ./build/isr.o ./build/irqstub.o ./build/isrstub.o ./build/clock.o -o ./build/kernel.elf

./build/kernel.bin: ./build/kernel.elf
	objcopy -O binary ./build/kernel.elf ./build/kernel.bin

./build/os.img: ./build/boot.bin ./build/kernel.bin
	cat ./build/boot.bin ./build/kernel.bin > ./build/os.img

clean:
	rm ./build/*.o ./build/*.bin ./build/*.img
run:
	qemu-system-i386 -drive format=raw,file=./build/os.img
