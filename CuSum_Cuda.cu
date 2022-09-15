#include "CuSum_Cuda.hpp"

__device__ double cuda_SQRT_2 = 1.414213562373095048801688724209698078569672;

__device__ double cuda_rozkl_norm_upr(double x){
    if(x>0)
        return erf(x/cuda_SQRT_2);
    else
        return -erf(-x/cuda_SQRT_2);
}

__global__ void cuda_CuSum(uint8_t **arrays, size_t *tab_numElements, double * P_values, int num_datas){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < num_datas){
        double *P_value = &(P_values[id]);
        uint8_t * array = arrays[id];
        uint64_t numElements = tab_numElements[id];

        int64_t S=0, _max=0, _min=0;
        uint8_t temp;
        // przejście po wszystkich bajtach
        for(uint64_t index = 0; index<numElements;index++){
            // przejscie po bitach
            temp = array[index];
            for(int i=0; i<8; i++){
                if(temp & 0x80) {
                    S++;
                    if(S > _max) _max=S;
                }
                else {
                    S--;
                    if(S < _min) _min=S;
                }
                temp <<= 1;
            }
        }
        int64_t z = (_max > -_min) ? _max : -_min;
        int64_t n = numElements*8;

        double sqrtNz = z / sqrt(n);
        double sum=0;

        /////////////////////////////////////////////
        //uproszczone pętle sum + upr rozkład normalny
        for(int64_t k = (-n/z-3)/4*4; k<-n/z+1; k+=4){
            sum-=cuda_rozkl_norm_upr((k+3) * sqrtNz);
            sum+=cuda_rozkl_norm_upr((k+1) * sqrtNz);
        }
        for(int64_t k = (-n/z+1)/4*4; k<=n/z-1; k+=4){
            sum += cuda_rozkl_norm_upr((k+1) * sqrtNz) * 2;
            sum -= cuda_rozkl_norm_upr((k-1) * sqrtNz);
            sum -= cuda_rozkl_norm_upr((k+3) * sqrtNz);
        }

        (*P_value) = 1-sum/2;
    }
}

// To dziala na tablicy dancyh i dzieli go na bloki o rozmiarze numElements
__global__ void cuda_CuSum_v2(uint8_t *arrays, size_t numElements, double * P_values, int num_datas){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < num_datas){
        double * P_value = &(P_values[id]);
        uint8_t * array = &arrays[numElements*id];

        int64_t S=0, _max=0, _min=0;
        uint8_t temp;
        // przejście po wszystkich bajtach
        for(uint64_t index = 0; index<numElements;index++){
            // przejscie po bitach
            temp = array[index];
            for(int i=0; i<8; i++){
                if(temp & 0x80) {
                    S++;
                    if(S > _max) _max=S;
                }
                else {
                    S--;
                    if(S < _min) _min=S;
                }
                temp <<= 1;
            }
        }
        int64_t z = (_max > -_min) ? _max : -_min;
        int64_t n = numElements*8;

        double sqrtNz = z / sqrt(n);
        double sum=0;

        /////////////////////////////////////////////
        //uproszczone pętle sum + upr rozkład normalny
        for(int64_t k = (-n/z-3)/4*4; k<-n/z+1; k+=4){
            sum-=cuda_rozkl_norm_upr((k+3) * sqrtNz);
            sum+=cuda_rozkl_norm_upr((k+1) * sqrtNz);
        }
        for(int64_t k = (-n/z+1)/4*4; k<=n/z-1; k+=4){
            sum += cuda_rozkl_norm_upr((k+1) * sqrtNz) * 2;
            sum -= cuda_rozkl_norm_upr((k-1) * sqrtNz);
            sum -= cuda_rozkl_norm_upr((k+3) * sqrtNz);
        }
        (*P_value) = 1-sum/2;
    }
}