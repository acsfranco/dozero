#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<time.h>
#include "../utils.h"

typedef struct {
  uint32_t num_images;
  uint32_t rows;
  uint32_t cols;
  uint8_t *data;
} MNIST_Images;

typedef struct {
  uint32_t num_labels;
  uint8_t *labels;
} MNIST_Labels;

uint32_t read_int_big_endian(FILE *f) {
  uint8_t bytes[4];
  fread(bytes, 1, 4, f);

  return (bytes[0] << 24) |
         (bytes[1] << 16) |
         (bytes[2] << 8)  |
         (bytes[3]);
}

MNIST_Images load_mnist_images(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    perror("Erro ao abrir arquivo de imagens!\n");
    exit(1);
  }

  uint32_t magic = read_int_big_endian(f);
  if (magic != 2051) {
    printf("Não é um arquivo MNIST válido!\n");
    exit(1);
  }

  MNIST_Images images;
  images.num_images = read_int_big_endian(f);
  images.rows       = read_int_big_endian(f);
  images.cols       = read_int_big_endian(f);

  uint32_t size = images.num_images * images.rows * images.cols;
  images.data = malloc(size);

  fread(images.data, 1, size, f);
  fclose(f);

  return images;
}

MNIST_Labels load_mnist_labels(const char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    perror("Erro ao abrir arquivo de rótulos");
    exit(1);
  }

  int magic = read_int_big_endian(f);
  if (magic != 2049) {
    printf("Magic number inválido: %d\n", magic);
    exit(1);
  }

  MNIST_Labels labels;
  labels.num_labels = read_int_big_endian(f);
  labels.labels = malloc(labels.num_labels);

  fread(labels.labels, 1, labels.num_labels, f);
  fclose(f);

  return labels;
}

float **convertmnisttodatainp(MNIST_Images images) {
  float **data = mallocmatrix(images.num_images, images.rows * images.cols);
  
  uint32_t datasize = images.rows * images.cols;
  for (uint32_t i = 0; i < images.num_images; i++) {
    for (uint32_t j = 0; j < datasize; j++) {
      data[i][j] = (float) images.data[datasize * i + j];
    }
  }

  return data;
}

float **convertmnisttodataout(MNIST_Labels labels) {
  float **data = mallocmatrix(labels.num_labels, 10);
  
  for (uint32_t i = 0; i < labels.num_labels; i++) {
    for (uint8_t j = 0; j < 10; j++) {
      data[i][j] = labels.labels[i] == j ? 1.0f : 0.0f; // oneshot out
    }
  }

  return data;
}

uint8_t **convertmnisttoimage(MNIST_Images images, uint32_t imageindex) {
  uint32_t datasize = images.rows * images.cols;
  uint8_t **imagedata = (uint8_t **) mallocmatrix(images.rows, images.cols);
  
  for (uint32_t i = 0; i < images.rows; i++) {
    for (uint32_t j = 0; j < images.cols; j++) {
      imagedata[i][j] = images.data[imageindex * datasize + i * images.rows + j];
    }
  }

  return imagedata;
}

/*int main() {
  srand(time(NULL));
  MNIST_Images images = load_mnist_images("train-images.idx3-ubyte");
  MNIST_Labels labels = load_mnist_labels("train-labels.idx1-ubyte");

  uint8_t **image = convertmnisttoimage(images, 10);
  float **datainp = convertmnisttodatainp(images);
  float **dataout = convertmnisttodataout(labels);
  
  printf("Imagens: %d\n", images.num_images);
  printf("Dimensão: %dx%d\n", images.rows, images.cols);
  printf("Primeiro rótulo: %d\n", labels.labels[0]);
  
  uint32_t *indexes = (uint32_t *)malloc(sizeof(uint32_t) * images.num_images);
  for (uint32_t i = 0; i < images.num_images; i++)
    indexes[i] = i;
  shuffle(indexes, images.num_images);
  uint32_t z;
 
  // Mostrar alguns pixels da primeira imagem
  for (int k = 0; k < 5; k++) {
    z = indexes[k];
    printf("Rótulo: %d\n", labels.labels[z]);
    
    for (int n = 0; n < 10; n++) {
      printf("Saida %d: %f\n", n, dataout[z][n]);  
    }
    
    for (int i = 0; i < 28; i++) {
      for (int j = 0; j < 28; j++) {
        uint8_t pixel = (uint8_t) datainp[z][i * images.rows + j];
        printf(pixel > 128 ? "#" : ".");
      }
      printf("\n");
    }
  }

  free(datainp);
  free(image);
  free(images.data);
  free(labels.labels);
  return 0;
}*/
