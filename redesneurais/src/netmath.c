#include <math.h>
#include <stdint.h>
#include "netmath.h"

/*
 * Calcula a função de custo Identidade.
 *
 * Parâmetros:
 *   x - Um valor escalar
 *
 * Retorno:
 *   valor do custo da rede neural
 */

float ident(float x) {
  return x;
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

float mse(float *out_true, float *out_pred, uint32_t samplesize) {
  float s = 0;
  for (uint32_t i = 0; i < samplesize; i++) {
    s += pow(out_pred[i] - out_true[i], 2);
  }
  s /= samplesize;
  return s;
}

