#include "neuron.h"

#ifndef NEURALNET_H
#define NEURALNET_H

/****************************
 * Estrutura da rede neural *
 * **************************/

typedef struct {
  NEURON *outneurons;
  uint32_t nout;
  float (*intactfunc)(float);
  float (*outactfunc)(float);
} NET;

/*
 * Computa o custo da rede neural
 *
 * Parâmetros:
 *   net - rede neural
 *   x - entradas das amostras
 *   y - saídas das amostras
 *   cost - a função de custo utilizada para calcular o custo
 *   samplesize - quantidade de amostras
 */

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize);

/*
 * Atualiza os parâmetros da rede em função dos gradientes
 *
 * Parâmetros
 *   net - rede neural
 *   neuron - camada de neurônios que serão atualizados os parâmetros
 *   cost - função de custo
 *   x - amostras de entrada da rede
 *   samplesize - quantidade de amostras
 */

void updateparams(NET *net, NEURON *neuron, float (*cost)(), float **x, float **y, uint32_t samplesize);

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

float computgradient(NET *net, float (*cost)(), float **x, float **y, float *param, uint32_t samplesize);

/*
 * Calcula o valor de saída de cada neurônio de saída da rede, dadas as entradas de uma amostra
 *
 * Parâmetros:
 *   net - rede neural 
 *   x - entradas de uma amostra
 *
 * Retorno:
 *   O valor de saída de cada neurônio de saída da rede
 */

float *feedforward(NET net, float *x);

/*
 * Cria uma rede neural MLP suas camadas e neurônios, inicializa seus pesos e bias aleatoriamente
 *
 * Parâmetros:
 *   layers - um vetor que informa a quantidade de neurônios de cada camada
 *   nlayers - o número de camadas da rede neural
 *   intactfunc - a função de ativação dos neurônios das camadas intermediárias
 *   outactfunc - a função de ativação dos neurônios da camada de saída da rede neural
 *
 * Retorno
 *   O rede neural MLP criada.
 */

NET initnet(uint32_t *layers, uint32_t nlayers, float (*intactfunc)(float), float (*outactfunc)(float));

/*
 * Função de treinamento da rede neural
 *
 * Parâmetros:
 * net - rede neural
 * cost - função de custo
 * x - entradas das amostras
 * y - saídas das amostras
 * samplesize - quantidade de amostras
 */

float train(NET *net, float (*cost)(), float **x, float **y, float samplesize);

#endif
