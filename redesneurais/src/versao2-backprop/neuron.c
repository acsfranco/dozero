#include "utils.h"
#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Computa o valor de saída do neurônio.
 *
 * Parâmetros:
 *   neuron - neurônio a ser computado
 *   x - vetor de entrada
 *
 * Retorno:
 *   Valor de saída do neurônio
 */

float computout(NEURON *neuron, float *x) {
  float k = 0;
  if (neuron->conneurons != NULL) {
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
      NEURON *cneuron = &neuron->conneurons[i];
      float a = computout(cneuron, x);
      cneuron->a = a;
      k += a * neuron->weights[i];
    } 
  } else {  
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
       k += x[i] * neuron->weights[i];
    }
  }
  k += neuron->bias;
  neuron->z = k; 
  return neuron->actfunc.func(k);
}

/*
 * Salva os pesos de uma rede em um arquivo
 *
 * Parâmetros:
 *   neurons - neurônios de uma camada da rede (para salvar os pesos de todos os neurônios, essa função deve ser chamada, passando os neurônios da camada de saída da rede.
 *   nneurons - número de neurônios da camada de neurônios
 *   f - ponteiro para o arquivo a ser salvo
 *   filename - nome do arquivo a ser salvo
 */

void saveweights(NEURON * neurons, uint32_t nneurons, FILE *f, const char *filename) {
  uint8_t open = 0;
  if (f == NULL) {
    f = fopen(filename, "wb");
    open = 1;
  }
  if (neurons[0].conneurons != NULL) {
    saveweights(neurons[0].conneurons, neurons[0].nconnections, f, "");
  }

  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    for (uint32_t j = 0; j < neuron->nconnections; j++) {
      fwrite(&neuron->weights[j], 4, 1, f);
    }
    fwrite(&neuron->bias, 4, 1, f);
  }

  if (open) {
    fclose(f);
  }
}

/*
 * Carrega, de um arquivo, os pesos de uma rede
 *
 * Parâmetros:
 *   neurons - neurônios de uma camada da rede (para carregar os pesos de todos os neurônios, essa função deve ser chamada, passando os neurônios da camada de saída da rede.
 *   nneurons - número de neurônios da camada de neurônios
 *   f - ponteiro para o arquivo a ser salvo
 *   filename - nome do arquivo a ser salvo
 */

void loadweights(NEURON * neurons, uint32_t nneurons, FILE *f, const char *filename) {
  uint8_t open = 0;
  if (f == NULL) {
    f = fopen(filename, "rb");
    open = 1;
  }
  if (neurons[0].conneurons != NULL) {
    loadweights(neurons[0].conneurons, neurons[0].nconnections, f, "");
  }

  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    for (uint32_t j = 0; j < neuron->nconnections; j++) {
      fread(&neuron->weights[j], 4, 1, f);
    }
    fread(&neuron->bias, 4, 1, f);
  }

  if (open) {
    fclose(f);
  }
}
