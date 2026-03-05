#include<stdint.h>

#ifndef MNIST_H
#define MNIST_H

/******************************************
 * Estrutura dos arquivos de imagem mnist *
 * ****************************************/

struct mnist_images {
  uint32_t magicnumber;
  uint32_t samplesize;
  uint32_t xsize;
  uint32_t ysize;
  uint8_t *images;
};

typedef struct mnist_images MNIST_Images;

/******************************************
 * Estrutura dos labels das imagens mnist *
 * ****************************************/

struct mnist_labels {
  uint32_t magicnumber;
  uint32_t samplesize;
  uint8_t *labels;
};

typedef struct mnist_labels MNIST_Labels;

/*
 * Converte um conjunto de bytes em Big Endian para um inteiro
 *
 * Parâmetros:
 *   bytes - vetor de bytes
 *
 * Retorno
 *   número inteiro convertido
 */

uint32_t convertBigEndianToInt(uint8_t bytes[4]);

/*
 * Carrega um arquivo do mnist, contendo as imagens dos dígitos
 *
 * Parâmetros:
 *   filaname - arquivo do mnist
 *
 * Retorno:
 *   Uma variável do tipo estruturado MNIST_Images, contendo os dados do arquivo
 */

MNIST_Images load_mnist_images(const char *filename);

/*
 * Carrega os arquivos de labels das imagens mnist
 *
 * Parâmetros:
 *   filaname - arquivo do mnist
 *
 * Retorno:
 *   Uma variável do tipo estruturado MNIST_Labels, contendo os dados do arquivo
 */

MNIST_Labels load_mnist_labels(const char *filename);

/*
 * Imprime, em modo texto, uma imagem do conjunto de imagens do mnist
 *
 * Parâmetros:
 *   images - uma variável do tipo estruturado MNIST_Images, contendo as
 *   informações do conjunto de dados
 *   index - o índice da imagem, no conjunto de dados, a ser impressa
 */

void print_MNIST_image(MNIST_Images images, uint32_t index);

/*
 * Converte o formato mnist no conjunto de entradas da rede neural
 *
 * Parâmetros:
 *   images - imagens do mnist
 *
 * Retorno
 *   conjunto de dados de entrada da rede neural
 */

float **convertmnistimagestoinp(MNIST_Images images);

/*
 * Converte os labels das imagens mnist no conjunto de saída da rede neural
 *
 * Parâmetros:
 *   labels - Os labels das imagens do mnist
 *
 * Retorno:
 *   conjunto de dados de saída da rede neural
 */

float **convertmnistlabelstoout(MNIST_Labels labels);

#endif
