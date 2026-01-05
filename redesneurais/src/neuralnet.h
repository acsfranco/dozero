#ifndef NEURALNET_H
#define NEURALNET_H

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

float computcost(NEURON neuron, float **x, float *y, float (*cost)(), uint32_t samplesize);

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

float computgradient(NEURON *neuron, float (*cost)(), float **x, float *y, float *param, uint32_t samplesize);

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

float train(NEURON *neuron, float (*cost)(), float **x, float *y, float samplesize);

#endif
