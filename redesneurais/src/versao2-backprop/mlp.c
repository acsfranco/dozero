/* mlp.c
 *
 * Implementação de uma mlp,
 * sem uso de bibliotecas externas.
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
  uint32_t layers[] = {784, 64, 10};
  ACTFUNC actsig = {sig, derivsig, 0};
  ACTFUNC actident = {ident, derivident, 1};
  NET net = initnet(layers, 3, actsig, actsig);
  
  MNIST_Images images;
  MNIST_Labels labels;

  images = load_mnist_images("mnist/train-images.idx3-ubyte");
  labels = load_mnist_labels("mnist/train-labels.idx1-ubyte");

  float **x = convertmnistimagestoinp(images);
  float **y = convertmnistlabelstoout(labels);

  float *y_hat = (float *) malloc(sizeof(float) * net.nout);
  float **xn = mallocmatrix(images.samplesize, 784);

  for (uint32_t i = 0; i < images.samplesize; i++) {
    for (uint32_t k = 0; k < 784; k++) {
      xn[i][k] = x[i][k] / 255;
    }
  }
  
  float **xb = mallocmatrix(64, 784);
  float **yb = mallocmatrix(64, 10);
  uint32_t *index = (uint32_t *)malloc(sizeof(uint32_t) * images.samplesize);
  for (uint32_t k = 0; k < images.samplesize; k++) {
    index[k] = k;
  } 
 
  for (uint32_t e = 0; e < EPOCHS; e++) {
    shuffle(index, images.samplesize);
    for (uint32_t s = 0; s < images.samplesize / BATCH; s++) {
      for (uint32_t b = 0; b < BATCH; b++) {
        xb[b] = xn[index[s * BATCH + b]];
        yb[b] = y[index[s * BATCH + b]];    
      }
      train(net, xb, yb, BATCH);
      if (s == 0) {
        float c = computcost(net, xb, yb, mse, BATCH);
        printf("EPOCA: %d custo: %f\n", e, c);
      }
    }
  }

  // Teste

  uint32_t hits = 0;

  images = load_mnist_images("mnist/t10k-images.idx3-ubyte");
  labels = load_mnist_labels("mnist/t10k-labels.idx1-ubyte");

  x = convertmnistimagestoinp(images);
  y = convertmnistlabelstoout(labels);

  for (uint32_t i = 0; i < images.samplesize; i++) {
    for (uint32_t k = 0; k < 784; k++) {
      xn[i][k] = x[i][k] / 255;
    }
  }
  
  for (uint32_t i = 0; i < images.samplesize; i++) {
    y_hat = feedforward(net, xn[i]);
    
    float max_prob = y_hat[0];
    uint8_t label_pred = 0;

    for (uint32_t k = 1; k < net.nout; k++) {
      if (y_hat[k] > max_prob) {
        max_prob = y_hat[k];
        label_pred = k;
      }
    }
    if (labels.labels[i] == label_pred) {
      hits++;
    }
    printf("Digito predito: %d Digito correto: %d Percentual de acerto: %.2f\n", label_pred, labels.labels[i], ((float) hits / (float)(i + 1)) * 100.0f); 
  }
}
