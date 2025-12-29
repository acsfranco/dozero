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
 * Computa a função de custo MSE - Mean of Squared Error
 *
 * Parâmetros:
 *   out_true - saídas das amostras de dados de treinamento
 *   out_pred - sáidas preditas pelo modelo
 *   samplesize - quantidade de amostras
 *
 * Retorno
 *   O cálculo do custo
 */

float mse(float *out_true, float *out_pred, uint32_t samplesize) {
  float s = 0;
  for (uint32_t i = 0; i < samplesize; i++) {
    s += pow(out_pred[i] - out_true[i], 2);
  }
  s /= samplesize;
  return s;
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
  NEURON neuron = initneuron(ident, 1);
  
  float **x = (float **) malloc(sizeof(float *) * 4);
  for (uint32_t i = 0; i < 4; i++) {
    x[i] = (float *) malloc(sizeof(float));
  }
  float *out_true = (float *) malloc(sizeof(float) * 4);
  float *out_pred = (float *) malloc(sizeof(float) * 4);

  x[0][0] = 0;
  x[1][0] = 2;
  x[2][0] = 4;
  x[3][0] = 6;

  out_true[0] = 6;
  out_true[1] = 11;
  out_true[2] = 16;
  out_true[3] = 21;
  
  neuron.weights[0] = 2.5;
  neuron.bias = 6;

  for (uint32_t i = 0; i < 4; i++) {
    out_pred[i] = computout(neuron, x[i]);
  }

  printf("O valor de w é %f\n: ", neuron.weights[0]);
  printf("O valor do bias é %f\n: ", neuron.bias);

  printf("O custo do neurônio é: %f\n", mse(out_true, out_pred, 4));
}

