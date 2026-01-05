#include <stdint.h>
#ifndef NEURON_H
#define NEURON_H

/* =======================
 * Estrutura do neurônio
 * ======================= */

typedef struct Neuron NEURON;

struct Neuron {
  float *weights;
  uint32_t nconnections;
  float bias;
  float (*actfunc)(float x);
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

float computout(NEURON neuron, float *x);

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

NEURON initneuron(float (*actfunc)(float x), uint32_t nconnections);

#endif
