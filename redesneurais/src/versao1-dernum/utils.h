#include <stdint.h>

#ifndef UTILS_H
#define UTILS_H

/*
 * Escolhe um valor randômico entre dois valores
 *
 * Parâmetros:
 *   min - valor minimo a ser escolhido
 *   max - valor máximo a ser escolhido
 *
 * Retorno
 *   Um valor randômico entre min e max 
 */

float randomize(float min, float max);

/*
 * Aloca uma matriz
 *
 * Parâmetros:
 * linha - número de linhas da matriz
 * colunas - número de colunas da matriz
 */

float ** mallocmatrix(uint32_t linhas, uint32_t colunas);

#endif
