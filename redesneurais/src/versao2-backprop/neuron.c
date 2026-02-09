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
