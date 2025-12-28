/*
 * perceptron.c
 *
 * Implementação de um perceptron,
 * sem uso de bibliotecas externas.
 *
 * Este arquivo contém:
 * - definição da estrutura do perceptron
 * - função de custo e computação do neurônio
 *
 * Objetivo educacional: mostrar como tudo funciona "por baixo".
 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>

typedef struct Neuron NEURON;

/* =======================
 * Estrutura do perceptron
 * ======================= */

struct Neuron {
  float *weights;
  uint32_t nconnections;
  float bias;
  float (*actfunc)(float x);
};

/*
 * Calcula a função de custo Identidade.
 *
 * Parâmetros:
 *   x - Um valor escalar
 *
 * Retorno:
 *   valor do custo da rede neural
 */

float ident(float x) {
  return x;
}

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
    printf("%f\n",x[i] * neuron.weights[i]);

    k += x[i] * neuron.weights[i];
  }
  k += neuron.bias;
  return neuron.actfunc(k);
}

/*
 * Escolhe um valor randômico entre dois valores
 *
 * Parâmetros:
 *   min - valor minimo a ser escolhido
 *   max - valor máximo a ser escolhido
 *
 * Retorno
 *   Um valor randômico entre min e max 
 */

float randomize(float min, float max) {
  float num = min + ((float)rand() / (float)RAND_MAX) * (max - min);
  return num;
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

void main() {
  srand(time(NULL));
  NEURON neuron = initneuron(ident, 4);
  float *x = (float *) malloc(sizeof(float) * 4);

  x[0] = 10;
  x[1] = 6;
  x[2] = -8;
  x[3] = 5;

  printf("A saída do neurônio é: %f\n", computout(neuron, x));
}

