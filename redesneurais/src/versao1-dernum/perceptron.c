/* perceptron.c
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
#include<stdint.h>
#include<time.h>
#include "neuron.h"
#include "neuralnet.h"
#include "netmath.h"
#include "utils.h"

void showweights(NEURON *neurons, uint32_t nneurons) {
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON neuron = neurons[i];
    if (neuron.conneurons != NULL) {
      showweights(neuron.conneurons, neuron.nconnections);
      for (uint32_t k = 0; k < neuron.nconnections; k++) {
        printf("PESO: %f\n", neuron.weights[k]);
      }
      printf("BIAS: %f\n", neuron.bias);

    } else {
      for (uint32_t k = 0; k < neuron.nconnections; k++) {
        printf("PESO: %f\n", neuron.weights[k]);
      }
       printf("BIAS: %f\n", neuron.bias);
    }
  } 
}

void main() {
  srand(time(NULL));
  uint32_t layers[] = {1, 3, 1};
  NET net = initnet(layers, 3, relu, ident);
  
  float **out_true = mallocmatrix(6,1);
  float **x = mallocmatrix(6,1);
    
  x[0][0] = 30;
  x[1][0] = 60;
  x[2][0] = 90;
  x[3][0] = 40;
  x[4][0] = 70;
  x[5][0] = 100;
  
  /*x[0][1] = 80;
  x[1][1] = 50;
  x[2][1] = 70;
  x[3][1] = 30;
  x[4][1] = 40;
  x[5][1] = 90;*/

  out_true[0][0] = 9.5;
  out_true[1][0] = 5.2;
  out_true[2][0] = 7.8;
  out_true[3][0] = 6;
  out_true[4][0] = 5.5;
  out_true[5][0] = 10.2;

  float *out_pred = (float *)malloc(sizeof(float) * net.nout);
  float **xn = mallocmatrix(6,1);

  for (uint32_t i = 0; i < 6; i++) {
    xn[i][0] = normalize(x[i][0], x, 0, 6);
  }
  
  for (uint32_t i = 0; i < 6; i++) {
    out_pred = feedforward(net, xn[i]);
    printf("Entradas %f - Saida %f\n", x[i][0], out_pred[0]);
  }

  for (int k = 0; k < 50000; k++) {
    train(&net, mse, xn, out_true, 6); 
  }
  
  printf("\n\nDepois do treinamento\n\n");
  printf("Pesos e bias treinados\n");
  showweights(net.outneurons, net.nout);

  for (uint32_t i = 0; i < 6; i++) {
    out_pred = feedforward(net, xn[i]);
    printf("Entradas %f %f - Saida %f\n", x[i][0], x[i][1], out_pred[0]);
  }
}
