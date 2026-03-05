#include <stdint.h>
#include <stdio.h>
#ifndef NEURON_H
#define NEURON_H

/* =======================
 * Estrutura do neurônio
 * ======================= */

typedef struct Neuron NEURON;

typedef struct {
  float (*func)(float);
  float (*deriv)(float);
  uint8_t derinptype; // 0 - z 1 - a;
} ACTFUNC;


struct Neuron {
  float *weights;
  float *grad_w;
  NEURON *conneurons;
  uint32_t nconnections;
  float bias;
  float grad_b;
  float delta;
  float a;
  float z;
  ACTFUNC actfunc;
};

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

float computout(NEURON *neuron, float *x);

/*
 * Salva os pesos de uma rede em um arquivo
 *
 * Parâmetros:
 *   neurons - neurônios de uma camada da rede (para salvar os pesos de todos os neurônios, essa função deve ser chamada, passando os neurônios da camada de saída da rede.
 *   nneurons - número de neurônios da camada de neurônios
 *   f - ponteiro para o arquivo a ser salvo
 *   filename - nome do arquivo a ser salvo
 */

void saveweights(NEURON * neurons, uint32_t nneurons, FILE *f, const char *filename);

/*
 * Carrega, de um arquivo, os pesos de uma rede
 *
 * Parâmetros:
 *   neurons - neurônios de uma camada da rede (para carregar os pesos de todos os neurônios, essa função deve ser chamada, passando os neurônios da camada de saída da rede.
 *   nneurons - número de neurônios da camada de neurônios
 *   f - ponteiro para o arquivo a ser salvo
 *   filename - nome do arquivo a ser salvo
 */

void loadweights(NEURON * neurons, uint32_t nneurons, FILE *f, const char *filename);

#endif
