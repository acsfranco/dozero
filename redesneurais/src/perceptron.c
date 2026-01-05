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
