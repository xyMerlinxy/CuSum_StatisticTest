#include "CuSum_Cuda.hpp"

__device__ double cuda_SQRT_2 = 1.414213562373095048801688724209698078569672;

__device__ double cuda_rozkl_norm_upr(double x){
    return (x>0)? erf(x/cuda_SQRT_2) : -erf(-x/cuda_SQRT_2);
}

__global__ void cuda_CuSum(uint8_t **dataArray, int numBlock, size_t *blockSizeArray, double * P_values){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < numBlock){
        double *P_value = &(P_values[id]);
        uint8_t * data = dataArray[id];
        uint64_t blockSize = blockSizeArray[id];

        int64_t S=0, _max=0, _min=0;
        uint8_t temp;
        // przejście po wszystkich bajtach
        for(uint64_t index = 0; index<blockSize;index++){
            // przejscie po bitach
            temp = data[index];
            #pragma unroll(8)
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
        int64_t n = blockSize*8;

        double sqrtNz = z / sqrt(n);
        double sum=0;

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

__global__ void cuda_CuSum(uint8_t *ptrData,  int numBlock, size_t blockSize, double * P_values){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < numBlock){
        double * P_value = &(P_values[id]);
        uint8_t * data = &ptrData[blockSize*id];

        int64_t S=0, _max=0, _min=0;
        uint8_t temp;
        // przejście po wszystkich bajtach
        for(uint64_t index = 0; index<blockSize;index++){
            // przejscie po bitach
            temp = data[index];
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
        int64_t n = blockSize*8;

        double sqrtNz = z / sqrt(n);
        double sum=0;

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


__global__ void cuda_CuSum_v2(uint8_t *ptrData,  int numBlock, size_t blockSize, double * P_values){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < numBlock){
        double * P_value = &(P_values[id]);
        uint8_t * data = &ptrData[blockSize*id];

        int64_t S=0, _max=0, _min=0;
        uint8_t temp;
        // przejście po wszystkich bajtach
        for(uint64_t index = 0; index<blockSize;index++){
            // przejscie po bitach
            temp = data[index];
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
        int64_t n = blockSize*8;

        double sqrtNz = z / sqrt(n);
        double sum=0;

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