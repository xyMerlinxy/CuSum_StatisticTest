#ifndef CUSUM_CUDA_HPP
#define CUSUM_CUDA_HPP

#include <iostream>

__global__ void cuda_CuSum(uint8_t **arrays, size_t *tab_numElements, double * P_values, int num_datas);
__global__ void cuda_CuSum_v2(uint8_t *arrays, size_t numElements, double * P_values, int num_datas);

#endif