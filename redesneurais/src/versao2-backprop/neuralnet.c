#include <stdint.h>
#include <stdlib.h>
#include "neuron.h"
#include "neuralnet.h"
#include "utils.h"
#include <string.h>
#include <stdio.h>
#define LR 0.1

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

float computcost(NET net, float **x, float **y, float (*cost)(), uint32_t samplesize) {
  float **out_pred = (float **) malloc(sizeof(float *) * samplesize);
  
  for (uint32_t i = 0; i < samplesize; i++) {
    out_pred[i] = feedforward(net, x[i]);
  }
  
  /**************************************************/
  float c = cost(y, out_pred, samplesize, net.nout);
  for (uint32_t i = 0; i < samplesize; i++) {
    free(out_pred[i]);
  }
  free(out_pred);
  /*************************************************/

  return c;
}

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

float *feedforward(NET net, float *x) {
  float *out = (float *)malloc(sizeof(float) * net.nout);

  for (uint32_t i = 0; i < net.nout; i++) {
    NEURON *neuron = &net.outneurons[i];
    out[i] = computout(neuron, x);
    neuron->a = out[i];   
  }

  return out;
}

void reset_forward(NEURON *neurons, uint32_t nneurons) {
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON *neuron = &neurons[i];
    neuron->a = 0;
    neuron->z = 0;
    neuron->delta = 0;
  }
  if (neurons[0].conneurons != NULL) {
    reset_forward(neurons[0].conneurons,  neurons[0].nconnections);
  }
}

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
 * Cria um neurônio, inicializa seu pesos e bia
 *
 * Parâmetros:
 *   actfunc - a função de ativação do neurônio
 *   nconnections - número de conexões do neurônio
 *
 * Retorno
 *   O neurônio criado.
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
        neuron.weights[n] = randomize(-1.0,1.0);
      }
      neuron.bias = 0.1; //randomize(-1, 1);
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
 * Função de treinamento da rede neural
 *
 * Parâmetros:
 * neuron - neurônio de saída da rede
 * cost - função de custo
 * x - entradas das amostras
 * y - saídas das amostras
 * samplesize - quantidade de amostras
 */
 
float train(NET net, float **x, float **y, float samplesize) {
  reset_grad(net.outneurons, net.nout);
  for (uint32_t i = 0; i < samplesize; i++) {
    reset_forward(net.outneurons, net.nout);
    updatedeltagrad(net, net.outneurons, net.nout, x[i], y[i], NULL, 0);
  }
  updateparams(net.outneurons, net.nout, samplesize);
}
