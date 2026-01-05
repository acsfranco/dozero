#include<stdint.h>

#ifndef NETMATH_H
#define NETMATH_H

/*
 * Calcula a função de custo Identidade.
 *
 * Parâmetros:
 *   x - Um valor escalar
 *
 * Retorno:
 *   valor do custo da rede neural
 */

float ident(float x);

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

float mse(float *out_true, float *out_pred, uint32_t samplesize);

#endif
