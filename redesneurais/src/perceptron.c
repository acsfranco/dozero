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

/*
 * Aloca uma matriz
 *
 * Parâmetros:
 * linha - número de linhas da matriz
 * colunas - número de colunas da matriz
 */

float ** mallocmatrix(uint32_t linhas, uint32_t colunas) {
  float **x = (float **) malloc(sizeof(float *) * linhas);
  
  for (uint32_t i = 0; i < linhas; i++) {
    x[i] = (float *) malloc(sizeof(float) * colunas);
  }
  
  return x;
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
  NEURON neuron = initneuron(ident, 2);
  
  float *out_true = (float *) malloc(sizeof(float) * 5);
  float **x = mallocmatrix(5,2);
    
  x[0][0] = 0; x[0][1] = 0;
  x[1][0] = 2; x[1][1] = 15;
  x[2][0] = 8; x[2][1] = 3;
  x[3][0] = 14; x[3][1] = 18;
  x[4][0] = 20; x[4][1] = 1;

  out_true[0] = 5;
  out_true[1] = 41;
  out_true[2] = 35;
  out_true[3] = 83;
  out_true[4] = 67;
  
  //neuron.weights[0] = 2.5;
  //neuron.bias = 6;

  float c = computcost(neuron, x, out_true, mse, 5);

  printf("O valor de w1 é %f\n: ", neuron.weights[0]);
  printf("O valor de w2 é %f\n: ", neuron.weights[1]);
  printf("O valor do bias é %f\n: ", neuron.bias);

  printf("O custo do neurônio antes do treinamento é: %f\n", c);

  for (int i = 0; i < 50000; i++) {
    train(&neuron, mse, x, out_true, 5);
  }

  c = computcost(neuron, x, out_true, mse, 5);
  
  printf("O valor de w1 é %f\n: ", neuron.weights[0]);
  printf("O valor de w2 é %f\n: ", neuron.weights[1]);
  printf("O valor do bias é %f\n: ", neuron.bias);

  printf("O custo do neurônio depois do treinamento é: %f\n", c);

  for (uint32_t i = 0; i < 5; i++) {
    //printf("Entradais %f %f - Saida %f\n", x[i][0], x[i][1],  out_true[i]);
    printf("Entradas %f %f - Saida %f\n", x[i][0], x[i][1],  computout(neuron, x[i]));
  }

}

