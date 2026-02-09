#include <stdint.h>
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
#endif
