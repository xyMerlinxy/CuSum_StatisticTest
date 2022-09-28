#ifndef CUSUM_CUDA_HPP
#define CUSUM_CUDA_HPP

#include <iostream>

// Funkcja realizuje test CuSum blokach danych o różnej długości.
// dataArray - tablica zawierająca tablice z danymi
// numBlock - ilość bloków danych
// blockSizeArray - tablica rozmiarów bloku
// Wyznaczana wartość P-value zapisywana jest do tablicy P_values.
__global__ void cuda_CuSum(uint8_t **dataArray, int numBlock, size_t *blockSizeArray, double * P_values);
// Funkcja realizuje test CuSum blokach danych o tej samej długości.
// ptrData - wskaźnik na dane
// numBlock - ilość bloków danych
// blockSize - rozmiar bloku
// Wyznaczana wartość P-value zapisywana jest do tablicy P_values.
__global__ void cuda_CuSum(uint8_t *ptrData,  int numBlock, size_t blockSize, double * P_values);



// Funkcja realizuje test CuSum blokach danych o tej samej długości.
// ptrData - wskaźnik na dane
// numBlock - ilość bloków danych
// blockSize - rozmiar bloku
// Wyznaczana wartość P-value zapisywana jest do tablicy P_values.
__global__ void cuda_CuSum_v2(uint8_t *ptrData,  int numBlock, size_t blockSize, double * P_values);
#endif