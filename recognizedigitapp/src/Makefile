CC = gcc
TARGET = mdr
LIBS = `pkg-config --cflags --libs sdl2 SDL2_ttf`
SRCS = mdr.c ../../redesneurais/src/versao2-backprop/neuron.c  ../../redesneurais/src/versao2-backprop/neuralnet.c  ../../redesneurais/src/versao2-backprop/netmath.c  ../../redesneurais/src/versao2-backprop/utils.c
OBJS = $(SRCS:.c=.o)
all:$(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LIBS) -lm

%.o:%.c
	$(CC) -c $< -o $@

clean:
	rm *.o $(TARGET)
