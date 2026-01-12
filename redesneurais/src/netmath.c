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
  return 1 / (1 + exp(-x));
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

