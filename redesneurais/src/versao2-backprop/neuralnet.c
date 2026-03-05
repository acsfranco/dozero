#include <stdint.h>
#include <stdlib.h>
#include "neuron.h"
#include "neuralnet.h"
#include "utils.h"
#include <string.h>
#include <stdio.h>
#define LR 0.05 // Learning Rate

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

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize) {
  float **out_pred = (float **) malloc(sizeof(float *) * samplesize);
  
  for (uint32_t i = 0; i < samplesize; i++) {
    out_pred[i] = feedforward(net, x[i]);
  }
  
  float c = cost(y, out_pred, samplesize, net.nout);
  for (uint32_t i = 0; i < samplesize; i++) {
    free(out_pred[i]);
  }
  free(out_pred);

  return c;
}

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

void updatedeltagrad(NET net, NEURON *neurons, uint32_t nneurons, float *x, float *y, BACKPARAMS *params, uint32_t nparams) {  
  float *y_hat = NULL;
  if (params == NULL) { // Calculando as predições da rede
    y_hat = feedforward(net, x);
  }

  BACKPARAMS *backparams = (BACKPARAMS *)malloc(sizeof(BACKPARAMS) * nneurons);
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    float deriv = neuron->actfunc.derinptype ? neuron->actfunc.deriv(neuron->z) : neuron->actfunc.deriv(neuron->a);    
    if (params == NULL) { // Calcular os deltas e os gradientes da camada de saída
      float error = (y_hat[i] - y[i]);
      neuron->delta = error * deriv;
    } else {
      neuron->delta = 0;
      for (uint32_t k = 0; k < nparams; k++) {
        neuron->delta += params[k].weights[i] * params[k].delta;
      }
      neuron->delta *= deriv;
    }
    // O cálculo dos gradientes dos pesos e dos bias
    for (uint32_t k = 0; k < neuron->nconnections; k++) {
      float a = neuron->conneurons != NULL ? neuron->conneurons[k].a : x[k];
      neuron->grad_w[k] += a * neuron->delta;
    }
    neuron->grad_b += neuron->delta;
    backparams[i].weights = neuron->weights;
    backparams[i].delta = neuron->delta;
  }
  free(params);
  free(y_hat);
  if (neurons[0].conneurons != NULL) {
    updatedeltagrad(net, neurons[0].conneurons, neurons[0].nconnections, x, y, backparams, nneurons);
  }
}

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

float *feedforward(NET net, float *x) {
  float *out = (float *)malloc(sizeof(float) * net.nout);

  for (uint32_t i = 0; i < net.nout; i++) {
    NEURON *neuron = &net.outneurons[i];
    out[i] = computout(neuron, x);
    neuron->a = out[i];   
  }

  return out;
}

/*
 * Zera os campos a, z e delta de cada neurônio, onde a é a saída e z o cálculo do somatório das entradas multiplicadas pelos pesos.
 *
 * Parâmetros:
 *   neurons - neurônios, de uma camada, que serão zerados o a, o z e o delta.
 *   nneurons - quantidade de neurônios dessa camada
 */

void reset_forward(NEURON *neurons, uint32_t nneurons) {
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    neuron->a = 0;
    neuron->z = 0;
    neuron->delta = 0;
  }
  if (neurons[0].conneurons != NULL) {
    reset_forward(neurons[0].conneurons, neurons[0].nconnections);
  }
}

/*
 * Zera os gradiente dos pesos de todos os neurônios da rede
 *
 * Parâmetros:
 *   neurons - neurônios, de uma camada, que serão zerados os gradientes
 *   nneurons - número de neurônios dessa camada
 */

void reset_grad(NEURON *neurons, uint32_t nneurons) {
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    for (uint32_t k = 0; k < neuron->nconnections; k++) {
      neuron->grad_w[k] = 0;
    }
    neuron->grad_b = 0;
  }
  if (neurons[0].conneurons != NULL) {
    reset_grad(neurons[0].conneurons,  neurons[0].nconnections);
  }
}

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

NET initnet(uint32_t *layers, uint32_t nlayers, ACTFUNC intactfunc, ACTFUNC outactfunc) {
  NET net;
  NEURON *prevlayer = NULL;
  net.nout = layers[nlayers - 1];
  for (int k = 1; k < nlayers; k++) {
    uint32_t nconnections = layers[k - 1];
    uint32_t nneurons = layers[k];
    NEURON *currlayer = (NEURON *)malloc(sizeof(NEURON) * nneurons);
    for (int i = 0; i < nneurons; i++) {
      NEURON neuron;
      neuron.nconnections = nconnections;
      neuron.conneurons = prevlayer;
      neuron.weights = (float *)malloc(sizeof(float) * nconnections);
      neuron.grad_w = (float *)malloc(sizeof(float) * nconnections);
      for (int n = 0; n < nconnections; n++) {
        neuron.weights[n] = randomize(-0.1f, 0.1f);
      }
      neuron.bias = randomize(-0.1f, 0.1f);
      neuron.actfunc = k < nlayers - 1 ? intactfunc : outactfunc;
      currlayer[i] = neuron;
    }
    prevlayer = (NEURON *)malloc(sizeof(NEURON) * nneurons);
    memcpy(prevlayer, currlayer, sizeof(NEURON) * nneurons);    
  }
  net.outneurons = prevlayer;
  reset_forward(net.outneurons, net.nout);
  reset_grad(net.outneurons, net.nout);
  return net;
}

/*
 * Atualiza os parâmetros da rede em função das médias dos gradientes
 *
 * Parâmetros
 *   neurons - neurônios, de uma camada da rede, que terão seu pesos e bias atualizados
 *   nneurons - número de neurônios dessa camada
 *   samplesize - quantidade de amostras
 */

void updateparams(NEURON *neurons, uint32_t nneurons, uint32_t samplesize) {
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    for (uint32_t k = 0; k < neuron->nconnections; k++) {
      neuron->weights[k] -= LR * (neuron->grad_w[k] / samplesize); 
    }
    neuron->bias -= LR * (neuron->grad_b / samplesize);
  }
  if (neurons[0].conneurons != NULL) {
    updateparams(neurons[0].conneurons, neurons[0].nconnections, samplesize);
  }
}

/*
 * Realiza o treinamento de uma época ou de um batch de uma rede neural MLP
 *
 * Parâmetros:
 *   net - rede neural
 *   x - entradas de todas as amostras ou de um batch
 *   y - saídas das amostras
 *   samplesize - quantidade de amostras
 */
 
void train(NET net, float **x, float **y, float samplesize) {
  reset_grad(net.outneurons, net.nout);
  for (uint32_t i = 0; i < samplesize; i++) {
    reset_forward(net.outneurons, net.nout);
    updatedeltagrad(net, net.outneurons, net.nout, x[i], y[i], NULL, 0);
  }
  updateparams(net.outneurons, net.nout, samplesize);
}
