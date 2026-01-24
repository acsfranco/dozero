#include <stdint.h>
#include <stdlib.h>
#include "neuron.h"
#include "neuralnet.h"
#include "utils.h"
#include <string.h>

#include <stdio.h>

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

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize) {
  float **out_pred = (float **) malloc(sizeof(float *) * samplesize);
  
  for (uint32_t i = 0; i < samplesize; i++) {
    out_pred[i] = feedforward(net, x[i]);
  }

  return cost(y, out_pred, samplesize, net.nout);
}

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

float computgradient(NET *net, float (*cost)(), float **x, float **y, float *param, uint32_t samplesize) {
  //      lim           cost(param + delta param) - cost(param)
  //  delta para -> 0   ---------------------------------------
  //                                  delta param

  *param += 0.0001; // param + delta param
  float variationcost = computcost(*net, x, y, cost, samplesize);
  *param -= 0.0001; // param
  float normalcost = computcost(*net, x, y, cost, samplesize);
  
  float gradient = (variationcost - normalcost) / 0.0001;
  return gradient;
}

float *feedforward(NET net, float *x) {
  float *out = (float *)malloc(sizeof(float) * net.nout);

  for (uint32_t i = 0; i < net.nout; i++) {
    out[i] = computout(net.outneurons[i], x);
  }

  return out;
}

/*
 * Cria um neurônio, inicializa seu pesos e bia
 *
 * Parâmetros:
 *   actfunc - a função de ativação do neurônio
 *   nconnections - número de conexões do neurônio
 *
 * Retorno
 *   O neurônio criado.
 */

NET initnet(uint32_t *layers, uint32_t nlayers, float (*intactfunc)(float), float (*outactfunc)(float)) {
  NET net;
  NEURON *prevlayer = NULL;
  net.nout = layers[nlayers - 1];
  for (int k = 1; k < nlayers; k++) {
    uint32_t nconnections = layers[k - 1];
    uint32_t nneurons = layers[k];
    NEURON *currlayer = (NEURON *)malloc(sizeof(NEURON) * nneurons);
    for (int i = 0; i < nneurons; i++) {
      NEURON neuron;
      neuron.nconnections = nconnections;
      neuron.conneurons = prevlayer;
      neuron.weights = (float *)malloc(sizeof(float) * nconnections);
      for (int n = 0; n < nconnections; n++) {
        neuron.weights[n] = randomize(-1.0,1.0);
        printf("Peso - %f\n", neuron.weights[n]);
      }
      neuron.bias = 0.1; // Colocar depois entre -1 e 1
      neuron.actfunc = k < nlayers - 1 ? intactfunc : outactfunc;
      currlayer[i] = neuron;
    }
    prevlayer = (NEURON *)malloc(sizeof(NEURON) * nneurons);
    memcpy(prevlayer, currlayer, sizeof(NEURON) * nneurons);    
  }
  net.outneurons = prevlayer;
  return net;
}

void updateparams(NET *net, NEURON *neuron, float (*cost)(), float **x, float **y, uint32_t samplesize) {
  float gradient;

  for (uint32_t i = 0; i < neuron->nconnections; i++) {
    gradient = computgradient(net, cost, x, y, &neuron->weights[i], samplesize);
    neuron->weights[i] -= 0.01 * gradient; 
  }
  gradient = computgradient(net, cost, x, y, &neuron->bias, samplesize);
  neuron->bias -= 0.01 * gradient;

  if (neuron->conneurons != NULL) {
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
      updateparams(net, &neuron->conneurons[i], cost, x, y, samplesize);
    }
  }
}


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
 
float train(NET *net, float (*cost)(float **, float **, uint32_t), float **x, float **y, float samplesize) {
  uint32_t nout = net->nout;

  for (uint32_t i = 0; i < nout; i++) {
    updateparams(net, &net->outneurons[i], cost, x, y, samplesize);
  }
}

