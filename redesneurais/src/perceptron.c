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

void main() {
  srand(time(NULL));
  NEURON neuron = initneuron(sig, 2);
  
  float *out_true = (float *) malloc(sizeof(float) * 6);
  float **x = mallocmatrix(6,2);
    
  x[0][0] = 6;
  x[1][0] = 5;
  x[2][0] = 4;
  x[3][0] = 1;
  x[4][0] = 1;
  x[5][0] = 2;
  
  x[0][1] = 1;
  x[1][1] = 0;
  x[2][1] = 1;
  x[3][1] = 4;
  x[4][1] = 2;
  x[5][1] = 3;

  out_true[0] = 1;
  out_true[1] = 1;
  out_true[2] = 1;
  out_true[3] = 0;
  out_true[4] = 0;
  out_true[5] = 0;

  //neuron.weights[0] = 2.5;
  //neuron.bias = 6;

  float c = computcost(neuron, x, out_true, mse, 6);

  for (uint8_t i = 0; i < 2; i++)
    printf("O valor de w%d é %f\n: ", i + 1, neuron.weights[i]);
  printf("O valor do bias é %f\n: ", neuron.bias);

  printf("O custo do neurônio antes do treinamento é: %f\n", c);

  for (int i = 0; i < 50000; i++) {
    train(&neuron, mse, x, out_true, 6);
  }
  c = computcost(neuron, x, out_true, mse, 6);
  
  for (uint8_t i = 0; i < 2; i++)
    printf("O valor de w%d é %f\n: ", i + 1, neuron.weights[i]);
  printf("O valor do bias é %f\n: ", neuron.bias);

  printf("O custo do neurônio depois do treinamento é: %f\n", c);

  for (uint32_t i = 0; i < 6; i++) {
    //printf("Entradais %f %f - Saida %f\n", x[i][0], x[i][1],  out_true[i]);
    printf("Entradas %f %f - Saida %f\n", x[i][0], x[i][1],  computout(neuron, x[i]));
  }
}
