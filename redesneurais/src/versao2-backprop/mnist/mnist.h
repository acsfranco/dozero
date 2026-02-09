#include<stdint.h>

#ifndef MNIST_H
#define MNIST_H

struct mnist_images {
  uint32_t magicnumber;
  uint32_t samplesize;
  uint32_t xsize;
  uint32_t ysize;
  uint8_t *images;
};

typedef struct mnist_images MNIST_Images;

struct mnist_labels {
  uint32_t magicnumber;
  uint32_t samplesize;
  uint8_t *labels;
};

typedef struct mnist_labels MNIST_Labels;
uint32_t convertBigEndianToInt(uint8_t bytes[4]);
MNIST_Images load_mnist_images(const char *filename);
MNIST_Labels load_mnist_labels(const char *filename);
void print_MNIST_image(MNIST_Images images, uint32_t index);
float **convertmnistimagestoinp(MNIST_Images images);
float **convertmnistlabelstoout(MNIST_Labels labels);

#endif
