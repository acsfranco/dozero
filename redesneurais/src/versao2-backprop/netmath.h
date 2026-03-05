#include<stdint.h>

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

/*
 * Derivada da função Identidade.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   retorna 1
 */

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

/*
 * Derivada da função sigmoid.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   cálculo da derivada
 */

float derivsig(float x);

/*
 * Função de ativação Relu.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   cálculo da Relu
 */

float relu(float x);

/*
 * Derivada da função Relu.
 *
 * Parâmetros:
 *   x - Entrada da função
 *
 * Retorno:
 *   cálculo da derivada
 */

float derivrelu(float x);

/*
 * Computa a função de custo MSE - Mean of Squared Error
 *
 * Parâmetros:
 *   out_true - saídas das amostras de dados de treinamento
 *   out_pred - sáidas preditas pelo modelo
 *   samplesize - quantidade de amostras
 *   nout - número de neurônios da camada de saída da rede
 *
 * Retorno
 *   O cálculo do custo
 */

float mse(float **out_true, float **out_pred, uint32_t samplesize, uint32_t nout);

/*
 * Normaliza um valor, em relação a um conjunto de dados, baseado no valor máximo e mínimo e na entrada desse conjunto.
 *
 * Parâmetros:
 *   input - valor a ser normalizado
 *   x - conjunto de dados
 *   inpindex - índice que corresponde a entrada do conjunto de dados a ser normalizada
 *   samplesize - número de amostras do conjunto de dados
 *
 * Retorno:
 *   o valor normalizado
 */

float normalize(float input, float **x, uint32_t inpindex, uint32_t samplesize);

#endif
