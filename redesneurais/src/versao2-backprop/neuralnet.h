#include "neuron.h"

#ifndef NEURALNET_H
#define NEURALNET_H

#define EPOCHS 20
#define BATCH 64 

typedef struct {
  NEURON *outneurons;
  uint32_t nout;
  float (*intactfunc)(float);
  float (*outactfunc)(float);
} NET;

typedef struct {
  float *weights;
  float delta;
} BACKPARAMS;

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

void updatedeltagrad(NET net, NEURON *neurons, uint32_t nneurons, float *x, float *y, BACKPARAMS *params, uint32_t nparams);

void updateparams(NEURON *neuron, uint32_t nneurons, uint32_t samplesize);
//void updateparams(NET *net, NEURON *neuron, float (*cost)(), float **x, float **y, uint32_t samplesize);

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

void reset_forward(NEURON *neurons, uint32_t nneurons);

void reset_grad(NEURON *neurons, uint32_t nneurons);

NET initnet(uint32_t *layers, uint32_t nlayers, ACTFUNC intactfunc, ACTFUNC outactfunc);

float train(NET net, float **x, float **y, float samplesize);

#endif
