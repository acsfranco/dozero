#include <stdint.h>
#include <stdlib.h>
#include "neuron.h"
#include "neuralnet.h"

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

float computcost(NEURON neuron, float **x, float *y, float (*cost)(), uint32_t samplesize) {
  float *out_pred = (float *) malloc(sizeof(float) * samplesize);
  
  for (uint32_t i = 0; i < samplesize; i++) {
    out_pred[i] = computout(neuron, x[i]);
  }

  return cost(y, out_pred, samplesize);
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

float computgradient(NEURON *neuron, float (*cost)(), float **x, float *y, float *param, uint32_t samplesize) {
  //      lim           cost(param + delta param) - cost(param)
  //  delta para -> 0   ---------------------------------------
  //                                  delta param

  *param += 0.0001; // param + delta param
  float variationcost = computcost(*neuron, x, y, cost, samplesize);
  *param -= 0.0001; // param
  float normalcost = computcost(*neuron, x, y, cost, samplesize);
  
  float gradient = (variationcost - normalcost) / 0.0001;
  return gradient;
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

float train(NEURON *neuron, float (*cost)(), float **x, float *y, float samplesize) {
  float gradient;
  for (uint32_t i = 0; i < neuron->nconnections; i++) {
    gradient = computgradient(neuron, cost, x, y, &neuron->weights[i], samplesize);
    neuron->weights[i] -= 0.001 * gradient;
  }
  gradient = computgradient(neuron, cost, x, y, &neuron->bias, samplesize);
  neuron->bias -= 0.001 * gradient;
}

