#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<string.h>
#include "../utils.h"
#include "mnist.h"

/****************************************************************
 * BIBLIOTECA MNIST: Funções para carregar os arquivos do mnist *
 * e convertê-los na entrada da rede neural.                    *
 * **************************************************************/

/*
 * Converte um conjunto de bytes em Big Endian para um inteiro
 *
 * Parâmetros:
 *   bytes - vetor de bytes
 *
 * Retorno
 *   número inteiro convertido
 */

uint32_t convertBigEndianToInt(uint8_t bytes[4]) {
  return bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3];
}

/*
 * Carrega um arquivo do mnist, contendo as imagens dos dígitos
 *
 * Parâmetros:
 *   filaname - arquivo do mnist
 *
 * Retorno:
 *   Uma variável do tipo estruturado MNIST_Images, contendo os dados do arquivo
 */

MNIST_Images load_mnist_images(const char *filename) {
  FILE *f = fopen(filename, "rb");
  MNIST_Images images;
  uint8_t bytes[4];
  fread(bytes, 4, 1, f); // Read magicnumber
  images.magicnumber = convertBigEndianToInt(bytes);
  if (images.magicnumber != 2051) {
    printf("Arquivo MNIST inválido!\n");
    fclose(f);
    return images;
  }
  fread(bytes, 4, 1, f); // Read sample size
  images.samplesize = convertBigEndianToInt(bytes);
  fread(bytes, 4, 1, f); // Read x size
  images.xsize = convertBigEndianToInt(bytes);
  fread(bytes, 4, 1, f); // Read y size
  images.ysize = convertBigEndianToInt(bytes);
  uint32_t datasize = images.samplesize * images.xsize * images.ysize;
  images.images = (uint8_t *) malloc(datasize);
  fread(images.images, datasize, 1, f);
  fclose(f);
  return images;
}

/*
 * Carrega os arquivos de labels das imagens mnist
 *
 * Parâmetros:
 *   filaname - arquivo do mnist
 *
 * Retorno:
 *   Uma variável do tipo estruturado MNIST_Labels, contendo os dados do arquivo
 */

MNIST_Labels load_mnist_labels(const char *filename) {
  FILE *f = fopen(filename, "rb");
  MNIST_Labels labels;
  uint8_t bytes[4];
  fread(bytes, 4, 1, f); // Read magicnumber
  labels.magicnumber = convertBigEndianToInt(bytes);
  if (labels.magicnumber != 2049) {
    printf("Arquivo MNIST inválido!\n");
    fclose(f);
    return labels;
  }
  fread(bytes, 4, 1, f); // Read sample size
  labels.samplesize = convertBigEndianToInt(bytes);
  labels.labels = (uint8_t *) malloc(labels.samplesize);
  fread(labels.labels, labels.samplesize, 1, f);
  fclose(f);
  return labels; 
}

/*
 * Imprime, em modo texto, uma imagem do conjunto de imagens do mnist
 *
 * Parâmetros:
 *   images - uma variável do tipo estruturado MNIST_Images, contendo as
 *   informações do conjunto de dados
 *   index - o índice da imagem, no conjunto de dados, a ser impressa
 */

void print_MNIST_image(MNIST_Images images, uint32_t index) {
  uint32_t imagesize = images.xsize * images.ysize;
  uint8_t *image = (uint8_t *) malloc(imagesize);
  memcpy(image, &images.images[index * imagesize], imagesize);
  for(uint32_t i = 0; i < images.ysize; i++) {
    for(uint32_t j = 0; j < images.xsize; j++) {
      uint32_t pixel = image[i * images.xsize + j];
      if (pixel < 128)
        printf(".");
      else
        printf("#");
    }
    printf("\n");
  }
}

/*
 * Converte o formato mnist no conjunto de entradas da rede neural
 *
 * Parâmetros:
 *   images - imagens do mnist
 *
 * Retorno
 *   conjunto de dados de entrada da rede neural
 */

float **convertmnistimagestoinp(MNIST_Images images) {
  uint32_t datasize = images.xsize * images.ysize;
  float **x = mallocmatrix(images.samplesize, datasize);
  for (uint32_t i = 0; i < images.samplesize; i++) {
    for (uint32_t k = 0; k < datasize; k++) {
      x[i][k] = (float)images.images[i * datasize + k]; // Tinha trocado k por j e esquecido do ;
    }
  }
  return x;
}

/*
 * Converte os labels das imagens mnist no conjunto de saída da rede neural
 *
 * Parâmetros:
 *   labels - Os labels das imagens do mnist
 *
 * Retorno:
 *   conjunto de dados de saída da rede neural
 */

float **convertmnistlabelstoout(MNIST_Labels labels) {
  uint32_t samplesize = labels.samplesize;
  float **y = mallocmatrix(samplesize, 10); // Tinha trocado samplesize por datasize
  for (uint32_t i = 0; i < labels.samplesize; i++) {
    for (uint8_t k = 0; k < 10; k++) {
      y[i][k] = k == labels.labels[i] ? 1.0f : 0.0f;
    }
  }
  return y;
}

