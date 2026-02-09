#Compilado

CC = gcc
# Executavel
TARGET = mlp

# Arquivo fontes
SRCS = mlp.c utils.c netmath.c neuralnet.c neuron.c mnist/mnist.c

# Arquivos OBSs
OBJS = $(SRCS:.c=.o)

# Regra padrão
all:$(TARGET)

$(TARGET):$(OBJS)
	$(CC) $(OBJS) -lm -o $(TARGET)

%.o: %.c
	$(CC) -c $< -o $@

clean:
	rm *.o $(TARGET)
