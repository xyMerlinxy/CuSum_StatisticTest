#include <iostream>
// #include <string>
// #include <bitset>
// #include <math.h>

#include "io.hpp"
#include "CuSum_Cuda.hpp"

using namespace std;


int main(int argc, char *argv[]){


    // TODO -  dodać obsługę parametrów wywołania programów
    // Tymczasowo jest na sztywno zaczytanie danych z plików w folderze data
    
    // zaczytanie plików
    vector<string> tab_name = file_names("data",".bin");
    vector<vector<uint8_t>> tab_data = read_all_files(tab_name);

    int h_num_file = tab_data.size();


	size_t * h_tab_size = (size_t*)malloc(sizeof(size_t)*h_num_file) ; // tablica z długościami plików;
	cout << "Ilość zaczytanych plików: " << h_num_file << endl;

    // inicjalizacja i alokacja pamięci na hoście
	uint8_t **h_tab_data = (uint8_t**)malloc(sizeof(uint8_t*)*h_num_file);
	double *h_tab_Pvalue = (double *)malloc(sizeof(double)*h_num_file);

    // uzupełnienie wsakźników do danych i ich rozmiaru
    for(int i=0; i<h_num_file; i++){
        h_tab_data[i] = tab_data[i].data();
        h_tab_size[i] = tab_data[i].size();
    }

	uint8_t **d_tab_data; // przechowuje wskaźnik na tablicę wskaźników na dane w GPU
    size_t * d_tab_size;  // wskaźnik na tablicę z rozmiarami danych na GPU
	cudaMalloc((void **)&d_tab_data, sizeof(uint8_t*)*h_num_file); 
	cudaMalloc((void **)&d_tab_size, sizeof(size_t)*h_num_file);

    uint8_t ** h_tab_ptr_GPU_data = (uint8_t**)malloc(sizeof(uint8_t*)*h_num_file); // tablica ze wskaźnikami na dane w GPU
    // kopiowanie danych na GPU
    for(int i=0;i<h_num_file;i++){
        cudaMalloc((void **)&(h_tab_ptr_GPU_data[i]),h_tab_size[i]);
        cudaMemcpy(h_tab_ptr_GPU_data[i],h_tab_data[i],h_tab_size[i],cudaMemcpyHostToDevice);
    }

    // przekopiowanie tablicy ze wskaźnikami na dane w GPU do GPU
    cudaMemcpy(d_tab_data, h_tab_ptr_GPU_data, sizeof(uint8_t*)*h_num_file, cudaMemcpyHostToDevice);
    // kopiowanie tablicy z rozmiarami danych do GPU
    cudaMemcpy(d_tab_size, h_tab_size, sizeof(size_t*)*h_num_file, cudaMemcpyHostToDevice);
   
    int threadsPerBlock = 32;
    int blocksPerGrid = (h_num_file+threadsPerBlock-1)/threadsPerBlock;;


    // dane z plików są w tablicy tablic d_tab_data
    // rozmiar tablicy d_tab_data to h_num_file
    // rozmiar tablic w tablicy d_tab_data znajduje się w tablicy d_tab_size

    ///////////// CuSum ////////////////
    double *d_tab_Pvalue; // wskaźnik na tablicę z P_Value na GPU
	cudaMalloc((void **)&d_tab_Pvalue, sizeof(double)*h_num_file); 

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_tab_data, d_tab_size ,d_tab_Pvalue, h_num_file);

    // kopia P_value z GPU do CPU
    cudaMemcpy(h_tab_Pvalue, d_tab_Pvalue, sizeof(double)*h_num_file,cudaMemcpyDeviceToHost);
    

    for(int i=0;i<h_num_file;i++){
        cout <<tab_name[i] << "\tP_value: " << h_tab_Pvalue[i]<< "\n";
    }
    free(h_tab_Pvalue);

    /////////// common ending /////////////////
    free(h_tab_size);
    free(h_tab_data);


    for(int i=0;i<h_num_file;i++){
        cudaFree(h_tab_ptr_GPU_data[i]);
    }
    cudaFree(d_tab_data);
    cudaFree(d_tab_Pvalue);
    cudaFree(d_tab_size);
	
    free(h_tab_ptr_GPU_data);
    return 0;
}
