#include "neuron.h"

#ifndef NEURALNET_H
#define NEURALNET_H

#define EPOCHS 20 // Número de épocas para o treinamento
#define BATCH 64 // Quantidade de elementos de um batch para o treinamento

/****************************
 * Estrutura de um neurônio *
 ****************************/

typedef struct {
  NEURON *outneurons;
  uint32_t nout;
  float (*intactfunc)(float);
  float (*outactfunc)(float);
} NET;

/********************************************************************
 * Estrutura dos parâmetros utilizados no algoritmo backpropagation *
 ********************************************************************/

typedef struct {
  float *weights;
  float delta;
} BACKPARAMS;

/*
 * Computa o custo da rede neural
 *
 * Parâmetros:
 *   net - rede neural
 *   x - entradas das amostras
 *   y - saídas das amostras
 *   cost - a função de custo utilizada para calcular o custo
 *   samplesize - quantidade de amostras
 *
 * Retorno:
 *   Custo da rede neural.
 */

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize);

/*
 * Atualiza os deltas e gradientes da rede neural
 *
 * Parâmetros:
 *   net - rede neural
 *   neurons - neurônios, de uma camada, que serão computados os gradientes e deltas
 *   nneuros - quantidade de neurônios dessa camada
 *   x - entradas de uma amostra
 *   y - saidas de uma amostra
 *   params - vetor de parâmetros que serão passados para a camada anterior (delta e pesos da camada corrente)
 *   nparams - quantidade de parâmetros em params
 */

void updatedeltagrad(NET net, NEURON *neurons, uint32_t nneurons, float *x, float *y, BACKPARAMS *params, uint32_t nparams);

/*
 * Atualiza os parâmetros da rede em função das médias dos gradientes
 *
 * Parâmetros
 *   neurons - neurônios, de uma camada da rede, que terão seu pesos e bias atualizados
 *   nneurons - número de neurônios dessa camada
 */

void updateparams(NEURON *neuron, uint32_t nneurons, uint32_t samplesize);

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
 * Zera os campos a, z e delta de cada neurônio, onde a é a saída e z o cálculo do somatório das entradas multiplicadas pelos pesos.
 *
 * Parâmetros:
 *   neurons - neurônios, de uma camada, que serão zerados o a, o z e o delta.
 *   nneurons - quantidade de neurônios dessa camada
 */

void reset_forward(NEURON *neurons, uint32_t nneurons);

/*
 * Zera os gradiente dos pesos de todos os neurônios da rede
 *
 * Parâmetros:
 *   neurons - neurônios, de uma camada, que serão zerados os gradientes
 *   nneurons - número de neurônios dessa camada
 */


void reset_grad(NEURON *neurons, uint32_t nneurons);

/*
 * Cria uma rede neural MLP suas camadas e neurônios, inicializa seus pesos e bias aleatoriamente
 *
 * Parâmetros:
 *   layers - um vetor que informa a quantidade de neurônios de cada camada
 *   nlayers - o número de camadas da rede neural
 *   intactfunc - a função de ativação e sua derivada dos neurônios das camadas intermediárias
 *   outactfunc - a função de ativação e sua derivada dos neurônios da camada de saída da rede neural
 *
 * Retorno
 *   O rede neural MLP criada.
 */

NET initnet(uint32_t *layers, uint32_t nlayers, ACTFUNC intactfunc, ACTFUNC outactfunc);

/*
 * Realiza o treinamento de uma época ou de um batch de uma rede neural MLP
 *
 * Parâmetros:
 *   net - rede neural
 *   x - entradas de todas as amostras ou de um batch
 *   y - saídas das amostras
 *   samplesize - quantidade de amostras
 */

void train(NET net, float **x, float **y, float samplesize);

#endif
