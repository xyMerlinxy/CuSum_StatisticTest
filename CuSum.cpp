#include <math.h>

#include "CuSum.hpp"

double SQRT_2 = 1.414213562373095048801688724209698078569672;

// ta funkcja implementuje uproszczona funkjcę noramlną
// w celu zwiększenia szybkości wykonania niektore jej elementy zostały przesunięte do funkcji nadrzędnej
double rozkl_norm_upr(double x){
    if(x>0)
        return erf(x/SQRT_2);
    else
        return -erf(-x/SQRT_2);
}

double CuSum(uint8_t *array, uint64_t numElements){
    int64_t S=0, _max=0, _min=0;
    uint8_t temp;
    // przejście po wszystkich bajtach
    for(uint64_t index = 0; index<numElements;index++){
        // przejscie po bitach
        temp = array[index];
        // na GPU można tą pętlę rozwinąć
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

    for(int64_t k = (-n/z-3)/4*4; k<-n/z+1; k+=4){
        sum-=rozkl_norm_upr((k+3) * sqrtNz);
        sum+=rozkl_norm_upr((k+1) * sqrtNz);
    }
    for(int64_t k = (-n/z+1)/4*4; k<=n/z-1; k+=4){
        sum += rozkl_norm_upr((k+1) * sqrtNz) * 2;
        sum -= rozkl_norm_upr((k-1) * sqrtNz);
        sum -= rozkl_norm_upr((k+3) * sqrtNz);
    }
    double P_value = 1-sum/2;
    return P_value;
}
