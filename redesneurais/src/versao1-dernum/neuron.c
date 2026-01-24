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

float computout(NEURON neuron, float *x) {
  float k = 0;
  if (neuron.conneurons != NULL) {
    for (uint32_t i = 0; i < neuron.nconnections; i++) {
      k += computout(neuron.conneurons[i], x) * neuron.weights[i];
    } 
  } else {  
    for (uint32_t i = 0; i < neuron.nconnections; i++) {
       k += x[i] * neuron.weights[i];
    }
  }
  k += neuron.bias;

  return neuron.actfunc(k);
}
