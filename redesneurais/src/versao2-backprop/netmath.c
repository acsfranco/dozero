#include <math.h>
#include <stdint.h>
#include "netmath.h"

/*
 * Função de ativação Identidade.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   retorna x
 */

float ident(float x) {
  return x;
}

float derivident(float x) {
  return 1;
}
/*
 * Função de ativação Sigmoid.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   cálculo da sigmoid
 */

float sig(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float derivsig(float x) {
  return x * (1 - x);
}

float relu(float x) {
  return x > 0 ? x : 0.0f;
}

float derivrelu(float x) {
  return x > 0 ? 1.0f : 0.0f;
}

/*
 * Computa a função de custo MSE - Mean of Squared Error
 *
 * Parâmetros:
 *   out_true - saídas das amostras de dados de treinamento
 *   out_pred - sáidas preditas pelo modelo
 *   samplesize - quantidade de amostras
 *
 * Retorno
 *   O cálculo do custo
 */

float mse(float **out_true, float **out_pred, uint32_t samplesize, uint32_t nout) {
  float s = 0;
  for (uint32_t i = 0; i < samplesize; i++) {
    for (uint32_t k = 0; k < nout; k++) {
      s += pow(out_pred[i][k] - out_true[i][k], 2);
    }
  }
  s /= (float)(samplesize * nout);
  return s;
}

float normalize(float input, float **x, uint32_t inpindex, uint32_t samplesize) {
  float max = x[0][inpindex], min = x[0][inpindex];

  for (uint32_t i = 1; i < samplesize; i++) {
    if (x[i][inpindex] < min) {
      min = x[i][inpindex];
    }
    
    if (x[i][inpindex] > max) {
      max = x[i][inpindex];
    }
  }

  return (input - min) / (max - min);
}
