#include<stdint.h>
#include "neuron.h"

#ifndef NETMATH_H
#define NETMATH_H

/*
 * Função de ativação Identidade.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   retorna x
 */

float ident(float x);
float derivident(float x);
/*
 * Função de ativação Sigmoid.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   cálculo da sigmoid
 */

float sig(float x);
float derivsig(float x);
float relu(float x);
float derivrelu(float x);
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

float mse(float **out_true, float **out_pred, uint32_t samplesize, uint32_t nout);
float normalize(float input, float **x, uint32_t inpindex, uint32_t samplesize);
#endif
