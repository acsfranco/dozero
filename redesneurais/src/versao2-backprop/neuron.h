#include <stdint.h>
#ifndef NEURON_H
#define NEURON_H

/* =======================
 * Estrutura do neurônio
 * ======================= */

typedef struct {
  float (*func)(float);
  float (*deriv)(float);
  uint8_t derinptype; // 0 - x 1 - y
                      // Exemplo: a derivada da sigmoid é
                      // sig(x) * (1 - sig(x))
                      // Só que sig(x) já foi calculado e
                      // é a saída da função de ativação.
                      // Logo, basta eu passar a saída da
                      // função de ativação e não a entrada.
} ACTFUNC;

typedef struct Neuron NEURON;

struct Neuron {
  float *weights;
  float *grad_w;
  NEURON *conneurons;
  uint32_t nconnections;
  float bias;
  float grad_b;
  float delta;
  float z;
  float a;
  uint8_t valid;
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
