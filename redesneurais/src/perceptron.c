#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>

typedef struct Neuron NEURON;

struct Neuron {
  float *weights;
  uint32_t nconnections;
  float bias;
  float (*actfunc)(float x);
};

float ident(float x) {
  return x;
}

float computout(NEURON neuron, float *x) {
  float k = 0;
  for (uint32_t i = 0; i < neuron.nconnections; i++) {
    printf("%f\n",x[i] * neuron.weights[i]);

    k += x[i] * neuron.weights[i];
  }
  k += neuron.bias;
  return neuron.actfunc(k);
}

float randomize(float min, float max) {
  float num = min + ((float)rand() / (float)RAND_MAX) * (max - min);
  return num;
}

NEURON initneuron(float (*actfunc)(float x), uint32_t nconnections) {
  NEURON neuron;
  neuron.actfunc = actfunc;
  neuron.nconnections = nconnections;
  neuron.weights = (float *)malloc(sizeof(float) * nconnections);
  for (uint32_t i = 0; i < nconnections; i++) {
    neuron.weights[i] = randomize(-1,1);
  }
  neuron.bias = randomize(-1,1);
  return neuron;
}

void main() {
  srand(time(NULL));
  NEURON neuron = initneuron(ident, 4);
  float *x = (float *) malloc(sizeof(float) * 4);

  x[0] = 10;
  x[1] = 6;
  x[2] = -8;
  x[3] = 5;

  printf("A saída do neurônio é: %f\n", computout(neuron, x));
}

