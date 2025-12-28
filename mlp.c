#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdint.h>
#include<string.h>

#define DELTA 0.1f;
#define RATE 0.1f;

typedef struct Neuron NEURON;
typedef struct Net NET;

struct Neuron {
  float *weights;
  NEURON *conneurons;
  float nconnections;
  float bias;
  float (*actfunc)(float x);
};

struct Net {
  NEURON *outneurons;
  uint32_t nout;
  float (*intactfunc)();
  float (*outactfunc)();
};

float computout(NEURON *neuron, float *x) {
  float sum = 0;
  if (neuron->conneurons != NULL) {
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
      sum += computout(&neuron->conneurons[i], x) * neuron->weights[i];
    }
  } else {
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
      sum += x[i] * neuron->weights[i];
    }
  }
  return neuron->actfunc(sum + neuron->bias);
}

float sig(float x) {
  return 1 / (1 + exp(-x));
}

float ident(float x) {
  return x;
}

float ReLu(float x) {
  return x > 0 ? x : 0; 
}

float *mallocweights(uint32_t nweights, float *bias) {
  float *weights = (float *) malloc(sizeof(float) * nweights);

  for (uint32_t i = 0; i < nweights; i++) {
    weights[i] = -2.0f + ((float)rand()/(float)RAND_MAX) * (4.0f);
  }
  *bias = -2.0f + ((float)rand()/(float)RAND_MAX) * (4.0f);

  return weights;
}

float** mallocmatrix(uint32_t nlin, uint32_t ncol) {
  float **mem = (float **)malloc(sizeof(float *) * nlin);
  for (int i = 0; i < nlin; i++) {
    mem[i] = (float *)malloc(sizeof(float) * ncol);
  }
  return mem;
}

float mse(float **out_true, float **out_pred, uint32_t samplesize, uint32_t nout) {
  float error = 0;
  for (int i = 0; i < samplesize; i++) {
    for (int k = 0; k < nout; k++) {
      error += pow(out_true[i][k] - out_pred[i][k], 2);
    }
  }
  error /= (float)(samplesize * nout) ;
  return error;
}

float *feedforward(NET *net, float *inp_set) {
  float *out = (float *)malloc(sizeof(float) * net->nout);
  for (uint32_t i = 0; i < net->nout; i++) {
    out[i] = computout(&net->outneurons[i], inp_set);
  }
  return out;
}

float computcost(NET *net, float (*cost)(), float **inp_set, float **out_set, uint32_t samplesize) {
  float **out_pred = (float **) malloc(sizeof(float**) * samplesize);
  for (int j = 0; j < samplesize; j++) {
    out_pred[j] = feedforward(net, inp_set[j]);
  }

  return cost(out_set, out_pred, samplesize, net->nout);
}

float computgradient(NET *net, float fx, float *params, float (*cost)(), float **inp_set, float **out_set, uint32_t samplesize) {
  float fxplusdx, gradient;
  *params += 0.0001f; // x + dx
  fxplusdx = computcost(net, cost, inp_set, out_set, samplesize);
  *params -= 0.0001f; // x - d
  gradient = (fxplusdx - fx) / 0.0001f;
  
  return gradient;
}

void updateparams(NET *net, NEURON *neuron, float (*cost)(), float **inp_set, float **out_set, uint32_t samplesize, float *fx) {
  float gradient;

  for (int i = 0; i < neuron->nconnections; i++) {
    gradient = computgradient(net, *fx, &neuron->weights[i], cost, inp_set, out_set, samplesize);
    neuron->weights[i] -= 0.01f * gradient; // atualizando o peso
    *fx = computcost(net, cost, inp_set, out_set, samplesize);
  }
  gradient = computgradient(net, *fx, &neuron->bias, cost, inp_set, out_set, samplesize);
  neuron->bias -= 0.01f * gradient;
  *fx = computcost(net, cost, inp_set, out_set, samplesize);
  if (neuron->conneurons != NULL) {
    for (uint32_t i = 0; i < neuron->nconnections; i++) {
      updateparams(net, &neuron->conneurons[i], cost, inp_set, out_set, samplesize, fx);
    }
  }
}

void train(NET *net, float (*cost)(), float **inp_set, float **out_set, uint32_t samplesize) {
  float fx = computcost(net, cost, inp_set, out_set, samplesize);

  for (int k = 0; k < net->nout; k++) {
    NEURON *neuron = &net->outneurons[k];
    updateparams(net, neuron, cost, inp_set, out_set, samplesize, &fx);
  }
}

NET initnet(uint32_t *layers, uint32_t nlayers, float (*intactfunc)(float), float (*outactfunc)(float)) {
  NET net;
  NEURON *prevlayer = NULL;
  net.nout = layers[nlayers - 1];
  for (int k = 1; k < nlayers; k++) { // Criando as camadas de neurônios da rede
    uint32_t nconnections = layers[k - 1];
    uint32_t nneurons = layers[k];
    NEURON *currlayer = (NEURON *) malloc(sizeof(NEURON) * nneurons);
    for (int i = 0; i < nneurons; i++) { // Construindo os neurônios da camada de saída
      NEURON neuron;
      neuron.conneurons = prevlayer;
      neuron.nconnections = nconnections;
      neuron.weights = mallocweights(nconnections, &neuron.bias);
      neuron.actfunc = k < nlayers - 1 ? intactfunc : outactfunc;
      currlayer[i] = neuron;
    }
    prevlayer = (NEURON *) malloc(sizeof(NEURON) * nneurons);
    memcpy(prevlayer, currlayer, sizeof(NEURON) * nneurons);
  }
  net.outneurons = prevlayer;
  return net;
}

void showweights(NEURON *neurons, uint32_t nneurons) {
  for (uint32_t i = 0; i < nneurons; i++) {
    if (i == 0) {
      if (neurons[i].conneurons != NULL) {
        showweights(neurons[i].conneurons, neurons[i].nconnections);
        printf("------------------ NOVA CAMADA ---------------------\n");
        for (int j = 0; j < neurons[i].nconnections; j++) {
          printf("PESO: %f\n",neurons[i].weights[j]);
        }
        printf("BIAS: %f\n", neurons[i].bias);
      } else {
        printf("--------------- CAMADA DE ENTRADA ------------------\n");
        for (int j = 0; j < neurons[i].nconnections; j++) {
          
          printf("PESO: %f\n",neurons[i].weights[j]);
        }
        printf("BIAS: %f\n", neurons[i].bias);
      }
    } else {
      for (int j = 0; j < neurons[i].nconnections; j++) {
        printf("PESO: %f\n",neurons[i].weights[j]);
      }
      printf("BIAS: %f\n", neurons[i].bias);
    }
  }
}

void main() {
  uint32_t layers[] = {1, 4, 1};
  srand((unsigned)time(NULL));
  NET net = initnet(layers, 3, ReLu, ident);
  
  
  printf("%d\n",rand());

  float **input = mallocmatrix(7,1);
  input[0][0] = -3;
  input[1][0] = -2;
  input[2][0] = -1;
  input[3][0] =  0;
  input[4][0] =  1;
  input[5][0] =  2;
  input[6][0] =  3;

  float **output = mallocmatrix(7,1);
  output[0][0] = 9;
  output[1][0] = 4;
  output[2][0] = 1;
  output[3][0] = 0;
  output[4][0] = 1;
  output[5][0] = 4;
  output[6][0] = 9;
  
  /*float **input = mallocmatrix(4,2);
  input[0][0] = 0.; input[0][1] = 0.;
  input[1][0] = 0.; input[1][1] = 1.;
  input[2][0] = 1.; input[2][1] = 0.;
  input[3][0] = 1.; input[3][1] = 1.;

  float **output = mallocmatrix(4,1);
  output[0][0] = 0.;
  output[1][0] = 1.;
  output[2][0] = 1.;
  output[3][0] = 0.;
  */

  for (int i = 0; i < 10000; i++) {
    if (i % 1000 == 0) {
      printf("CUSTO: %f\n", computcost(&net, mse, input, output, 7));
    }
    train(&net, mse, input, output, 7);
  } 

  for (int i = 0; i < 7; i++) {
    float *out = feedforward(&net, input[i]);
    printf(" # %f -> %f\n", input[i][0], out[0]);
  }

  showweights(net.outneurons, 1);
}
