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
#include "mnist/mnist.h"

FILE *f;

void saveweights(NEURON *neurons, uint32_t nneurons) { /******************************************************/
  for (uint32_t i = 0; i < nneurons; i++) {
    NEURON neuron = neurons[i];
    if (neuron.conneurons != NULL && i == 0) {
      saveweights(neuron.conneurons, neuron.nconnections);
    }     
    for (uint32_t k = 0; k < neuron.nconnections; k++) {
      fwrite(&neuron.weights[k], sizeof(float), 1, f); 
    }
    fwrite(&neuron.bias, sizeof(float), 1, f); 
  }
}

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
  ACTFUNC actident = {ident, derivident, 1};
  ACTFUNC actsig = {sig, derivsig, 0};
  ACTFUNC actrelu = {relu, derivrelu, 1};

  srand(time(NULL));
  uint32_t layers[] = {784, 64, 10};
  NET net = initnet(layers, 3, actsig, actsig);
  
  
  MNIST_Images images = load_mnist_images("mnist/train-images.idx3-ubyte");
  MNIST_Labels labels = load_mnist_labels("mnist/train-labels.idx1-ubyte");

  float **x = convertmnisttodatainp(images);
  float **out_true = convertmnisttodataout(labels);
 
  float *out_pred = (float *)malloc(sizeof(float) * net.nout);
  float **xn = mallocmatrix(images.num_images, 784);
  
  for (uint32_t i = 0; i < images.num_images; i++) {
    for (uint32_t k = 0; k < 784; k++) {
      xn[i][k] = x[i][k] / 255;
    }
  }
  
  uint32_t *indexes = (uint32_t *)malloc(sizeof(uint32_t) * images.num_images);
  for (uint32_t i = 0; i < images.num_images; i++)
    indexes[i] = i;
  
  float **xb = (float **)malloc(sizeof(float *) * BATCH);
  float **ob = (float **)malloc(sizeof(float *) * BATCH);
  for (int k = 0; k < EPOCHS; k++) {
    shuffle(indexes, images.num_images);
    printf("EPOCA: %d\n",k);

    for (uint32_t b = 0; b < (uint32_t)images.num_images / BATCH; b++) {
      for(uint32_t i = 0; i < BATCH; i++){
        uint32_t z = indexes[b * BATCH + i];  
        xb[i] = xn[z];
        ob[i] = out_true[z];
      }
      
      train(net, xb, ob, 64);
      if (b == 0) {
        float c = computcost(net, xb, ob, mse, 64); 
        printf("EPOCH: %d Custo: %f\n", k, c);
      }
   }
  }

  free(xb);
  free(ob);

  printf("\n\nDepois do treinamento\n\n");
  printf("Pesos e bias treinados\n");
  
  images = load_mnist_images("mnist/t10k-images.idx3-ubyte");
  labels = load_mnist_labels("mnist/t10k-labels.idx1-ubyte");
  free(x);
  free(out_true);
  x = convertmnisttodatainp(images);
  out_true = convertmnisttodataout(labels);
  out_pred = (float *)malloc(sizeof(float) * net.nout);

  for (uint32_t i = 0; i < images.num_images; i++) {
    for (uint32_t k = 0; k < 784; k++) {
      xn[i][k] = x[i][k] / 255;
    }
  }
  
  printf("Salvando os pesos da rede...\n");
  f = fopen("neuralweights.net", "wb");
  saveweights(net.outneurons, net.nout);
  uint32_t hits = 0;
  for (uint32_t i = 0; i < 60; i++) {
    out_pred = feedforward(net, xn[i]);
    uint8_t dig_pred = 0, dig_true = 0;
    float maxprobdig = out_pred[0];
    
    for (uint32_t n = 1; n < net.nout; n++) {
      if (maxprobdig < out_pred[n]){
        maxprobdig = out_pred[n];
        dig_pred = n;
      }
    }
    if (labels.labels[i] == dig_pred)
      hits++;
    printf("Digito predito pela NLP: %d Digito correto: %d\n",dig_pred,labels.labels[i]);
    printf("Percetual de acerto: %.2f\n", (float)hits / (float)(i + 1)); 
  }
  
  fclose(f); 
  free(images.data);
  free(labels.labels);
  free(x);
  free(out_true);
  free(out_pred);
  free(xn);
  free(indexes);

/*  NET net = initnet(layers, 3, actrelu, actident);
  float **out_true = mallocmatrix(6,1);
  float **x = mallocmatrix(6,1);
    
  x[0][0] = 30;
  x[1][0] = 60;
  x[2][0] = 90;
  x[3][0] = 40;
  x[4][0] = 70;
  x[5][0] = 100;
  
  x[0][1] = 80;
  x[1][1] = 50;
  x[2][1] = 70;
  x[3][1] = 30;
  x[4][1] = 40;
  x[5][1] = 90;

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
    //xn[i][1] = normalize(x[i][1], x, 1, 6);
  }
  
  for (uint32_t i = 0; i < 6; i++) {
    initparams(net.outneurons, net.nout);
    out_pred = feedforward(net, xn[i]);
    printf("Entradas %f - Saida %f\n", x[i][0], out_pred[0]);
  }

  for (int k = 0; k < 300000; k++) {
    train(net, xn, out_true, 6); 
  }
  
  printf("\n\nDepois do treinamento\n\n");
  printf("Pesos e bias treinados\n");
  showweights(net.outneurons, net.nout);

  for (uint32_t i = 0; i < 6; i++) {
    initparams(net.outneurons, net.nout);
    out_pred = feedforward(net, xn[i]);
    printf("Entradas %f - Saida %f\n", x[i][0], out_pred[0]);
  }*/
}
