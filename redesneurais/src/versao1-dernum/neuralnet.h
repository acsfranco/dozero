#include "neuron.h"

#ifndef NEURALNET_H
#define NEURALNET_H

typedef struct {
  NEURON *outneurons;
  uint32_t nout;
  float (*intactfunc)(float);
  float (*outactfunc)(float);
} NET;

/*
 * Computa o custo da rede neural
 *
 * Parâmetros:
 *   neuron - neurônio de saída da rede ou o perceptron a ser calculado o custo
 *   x - entradas das amostras
 *   y - saídas das amostras
 *   cost - a função de custo utilizada para calcular o custo
 *   samplesize - quantidade de amostras
 */

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize);
void updateparams(NET *net, NEURON *neuron, float (*cost)(), float **x, float **y, uint32_t samplesize);

/*
 * Calcula o gradiente pelo método da derivada numérica
 *
 * Parâmetros:
 *   neuron - endereço do neurônio com o parâmetro utilizado no cálculo
 *   cost - função de custo
 *   x - entradas das amostras
 *   y - saídas das amostras
 *   param - endereço do parâmetro utilizado no cálcuo
 *   samplesize - tamanho da amostra
 */

float computgradient(NET *net, float (*cost)(), float **x, float **y, float *param, uint32_t samplesize);

/*
 * Função de treinamento da rede neural
 *
 * Parâmetros:
 * neuron - neurônio de saída da rede
 * cost - função de custo
 * x - entradas das amostras
 * y - saídas das amostras
 * samplesize - quantidade de amostras
 */

float *feedforward(NET net, float *x);

NET initnet(uint32_t *layers, uint32_t nlayers, float (*intactfunc)(float), float (*outactfunc)(float));

float train(NET *net, float (*cost)(), float **x, float **y, float samplesize);

#endif
