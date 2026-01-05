#include "utils.h"
#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

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

float computout(NEURON neuron, float *x) {
  float k = 0;
  for (uint32_t i = 0; i < neuron.nconnections; i++) {
     k += x[i] * neuron.weights[i];
  }
  k += neuron.bias;
  return neuron.actfunc(k);
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

NEURON initneuron(float (*actfunc)(float x), uint32_t nconnections) {
  NEURON neuron;
  neuron.actfunc = actfunc;
  neuron.nconnections = nconnections;
  neuron.weights = (float *)malloc(sizeof(float) * nconnections);
  for (uint32_t i = 0; i < nconnections; i++) {
    neuron.weights[i] = randomize(-1,1);
  }
  neuron.bias = randomize(-1,1);
  return neuron;
}

