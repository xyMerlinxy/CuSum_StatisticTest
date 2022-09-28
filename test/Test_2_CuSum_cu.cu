#include <vector>

#include "./../io.hpp"
#include "./../CuSum_Cuda.hpp"

#include <filesystem>

using namespace std;


bool Test_2_CuSum_cu_v1(){
    double file_P_value = 0.451231;
    vector<uint8_t> file = read_file("test/test_data/data.sha1");
	size_t h_size = file.size();

	uint8_t *h_data = file.data();
	double h_Pvalue;

	uint8_t * d_data; // dane w GPU
    double *d_Pvalue; // wskaźnik na tablicę z P_Value na GPU

	cudaMalloc((void **)&d_data, sizeof(uint8_t)*h_size); 
	cudaMalloc((void **)&d_Pvalue, sizeof(double)); 

    cudaMemcpy(d_data, h_data, h_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = 1;

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, 1, h_size, d_Pvalue);

    cudaMemcpy(&h_Pvalue, d_Pvalue, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_Pvalue);

    return abs(file_P_value-h_Pvalue)<1e-5;
}

bool Test_2_CuSum_cu_v2(){
    double file_P_value = 0.547944;
    vector<uint8_t> file = read_file("test/test_data/data.bad_rng");
	size_t h_size = file.size();

	uint8_t *h_data = file.data();
	double h_Pvalue;

	uint8_t * d_data; // dane w GPU
    double *d_Pvalue; // wskaźnik na tablicę z P_Value na GPU

	cudaMalloc((void **)&d_data, sizeof(uint8_t)*h_size); 
	cudaMalloc((void **)&d_Pvalue, sizeof(double)); 

    cudaMemcpy(d_data, h_data, h_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = 1;

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, 1, h_size, d_Pvalue);

    cudaMemcpy(&h_Pvalue, d_Pvalue, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_Pvalue);

    return abs(file_P_value-h_Pvalue)<1e-5;
}
bool Test_2_CuSum_cu_v3(){
    double file_P_value = 0.405915;
    vector<uint8_t> file = read_file("test/test_data/100_1_0.bin");
	size_t h_size = file.size();

	uint8_t *h_data = file.data();
	double h_Pvalue;

	uint8_t * d_data; // dane w GPU
    double *d_Pvalue; // wskaźnik na tablicę z P_Value na GPU

	cudaMalloc((void **)&d_data, sizeof(uint8_t)*h_size); 
	cudaMalloc((void **)&d_Pvalue, sizeof(double)); 

    cudaMemcpy(d_data, h_data, h_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = 1;

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, 1, h_size, d_Pvalue);

    cudaMemcpy(&h_Pvalue, d_Pvalue, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_Pvalue);

    return abs(file_P_value-h_Pvalue)<1e-5;
}
bool Test_2_CuSum_cu_v4(){
    double file_P_value = 0.434313;
    vector<uint8_t> file = read_file("test/test_data/1000000_1_0.bin");
	size_t h_size = file.size();

	uint8_t *h_data = file.data();
	double h_Pvalue;

	uint8_t * d_data; // dane w GPU
    double *d_Pvalue; // wskaźnik na tablicę z P_Value na GPU

	cudaMalloc((void **)&d_data, sizeof(uint8_t)*h_size); 
	cudaMalloc((void **)&d_Pvalue, sizeof(double)); 

    cudaMemcpy(d_data, h_data, h_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = 1;

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, 1, h_size, d_Pvalue);

    cudaMemcpy(&h_Pvalue, d_Pvalue, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_Pvalue);

    return abs(file_P_value-h_Pvalue)<1e-5;
}


int Test_2(){
    if (Test_2_CuSum_cu_v1()){
        cout << "   Test CuSum.cu_1_1\tOK\n";
    }
    else{
        cout << "   Test CuSum.cu_1_1\tFAILED\n";
        return false;
    }

    if (Test_2_CuSum_cu_v2()){
        cout << "   Test CuSum.cu_1_2\tOK\n";
    }
    else{
        cout << "   Test CuSum.cu_1_2\tFAILED\n";
        return false;
    }

    if (Test_2_CuSum_cu_v3()){
        cout << "   Test CuSum.cu_1_3\tOK\n";
    }
    else{
        cout << "   Test CuSum.cu_1_3\tFAILED\n";
        return false;
    }

    if (Test_2_CuSum_cu_v4()){
        cout << "   Test CuSum.cu_1_4\tOK\n";
    }
    else{
        cout << "   Test CuSum.cu_1_4\tFAILED\n";
        return false;
    }

    return true;
}