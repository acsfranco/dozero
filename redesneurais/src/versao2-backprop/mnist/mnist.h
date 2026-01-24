#ifndef MNIST_H
#define MNIST_H
#include<stdint.h>

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

uint32_t read_int_big_endian(FILE *f);
MNIST_Images load_mnist_images(const char *filename);
MNIST_Labels load_mnist_labels(const char *filename);
float **convertmnisttodatainp(MNIST_Images images);
float **convertmnisttodataout(MNIST_Labels labels);
uint8_t **convertmnisttoimage(MNIST_Images images, uint32_t imageindex);

#endif
