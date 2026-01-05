#include <stdint.h>
#include <stdlib.h>
#include "utils.h"

/*
 * Escolhe um valor randômico entre dois valores
 *
 * Parâmetros:
 *   min - valor minimo a ser escolhido
 *   max - valor máximo a ser escolhido
 *
 * Retorno
 *   Um valor randômico entre min e max 
 */

float randomize(float min, float max) {
  float num = min + ((float)rand() / (float)RAND_MAX) * (max - min);
  return num;
}

/*
 * Aloca uma matriz
 *
 * Parâmetros:
 * linha - número de linhas da matriz
 * colunas - número de colunas da matriz
 */

float ** mallocmatrix(uint32_t linhas, uint32_t colunas) {
  float **x = (float **) malloc(sizeof(float *) * linhas);
  
  for (uint32_t i = 0; i < linhas; i++) {
    x[i] = (float *) malloc(sizeof(float) * colunas);
  }
  
  return x;
}
