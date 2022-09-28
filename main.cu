#include <iostream>
#include <chrono>
#include <fstream>

#include "io.hpp"
#include "CuSum.hpp"
#include "CuSum_Cuda.hpp"
#include "test/Test.hpp"

using namespace std;

// dane są w postaci wielu bloków o różnych rozmiarach
// dane są czytane z plików
void data_from_file(){
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

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_tab_data, h_num_file, d_tab_size, d_tab_Pvalue);

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

}

// dane są w postaci wielu bloków o tym samym rozmiarze
// dane są generowane
void many_block_many_size(){

    //pamięć podzielona na 1024 pliki
    int h_num_file = 1024*1024;
    int file_size = 1024;
    cout << "Ilość generowanych plików: " << h_num_file << endl;

    size_t * h_tab_size = (size_t*)malloc(sizeof(size_t)*h_num_file) ; // tablica z długościami plików;

    // inicjalizacja i alokacja pamięci na hoście
    uint8_t **h_tab_data = (uint8_t**)malloc(sizeof(uint8_t*)*h_num_file);
    double *h_tab_Pvalue = (double *)malloc(sizeof(double)*h_num_file);

    // generowanie plików
    for(int i=0; i<h_num_file; i++){
        h_tab_data[i] = (uint8_t *)malloc(file_size);
        h_tab_size[i] = file_size;
        for(int j = 0;j<file_size;j++){
            h_tab_data[i][j]=rand()%256;
        }
    }
    cout << "Wygenerowano " << h_num_file << " o rozmiarze " << file_size << "\n";

    //int threadsPerBlock = 32;
    int threadsPerBlock = 128;
    int blocksPerGrid = (h_num_file+threadsPerBlock-1)/threadsPerBlock;;
    cout << "ThreadsPerBlock:" << threadsPerBlock << endl; 

    auto GPU_t_start = std::chrono::high_resolution_clock::now();

    uint8_t **d_tab_data;
    double *d_tab_Pvalue;
    size_t * d_tab_size;
    cudaMalloc((void **)&d_tab_data, sizeof(uint8_t*)*h_num_file); 
    cudaMalloc((void **)&d_tab_Pvalue, sizeof(double)*h_num_file); 
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

    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_tab_data, h_num_file, d_tab_size, d_tab_Pvalue);

    cout << "GPU zakończyło obliczenia\n";

    cudaMemcpy(h_tab_Pvalue, d_tab_Pvalue, sizeof(double)*h_num_file,cudaMemcpyDeviceToHost);
    auto GPU_t_end = std::chrono::high_resolution_clock::now();
    double GPU_time = std::chrono::duration<double, std::milli>(GPU_t_end-GPU_t_start).count();

    ////////////////////////// CPU //////////////////////
    auto CPU_t_start = std::chrono::high_resolution_clock::now();
    double * CPU_P_value = (double *)malloc(sizeof(double)*h_num_file);
    for(int i=0;i<h_num_file;i++){
                                                     //CPU_P_value[i] = CuSum(h_tab_data[i],h_tab_size[i]);
        CPU_P_value[i] = CuSum(h_tab_data[i],h_tab_size[i]);
    }

    auto CPU_t_end = std::chrono::high_resolution_clock::now();
    double CPU_time = std::chrono::duration<double, std::milli>(CPU_t_end-CPU_t_start).count();

    cout << "CPU zakończyło obliczenia\n";

    ////////////////////////// check resoult /////////////
    // sprawdzenie poprawnosci obliczeń
    for(int i=0;i<h_num_file;i++){
        if (abs(h_tab_Pvalue[i] - CPU_P_value[i]) > 1e-5) {
            cout << h_tab_Pvalue[i] << " " << CPU_P_value[i] << "\n";
            fprintf(stderr, "Result verification failed at index %i!\n",i);
            exit(EXIT_FAILURE);
        }
    }

    // cout<<"Czas wykonania GPU: "<<milliseconds<< "\n";
    cout<<"Czas wykonania GPU: "<<GPU_time<< "\n";
    cout<<"Czas wykonania CPU: "<<CPU_time<< "\n";

    free(h_tab_size);
    free(h_tab_data);
    free(h_tab_Pvalue);

    for(int i=0;i<h_num_file;i++){
        cudaFree(h_tab_ptr_GPU_data[i]);
    }
    cudaFree(d_tab_data);
    cudaFree(d_tab_size);
    cudaFree(d_tab_Pvalue);
	
    free(h_tab_ptr_GPU_data);
}

// dane są w postaci jednego bloku danych i są dzielone na mniejsze bloki danych
// dane są generowane losowo
void one_block(){
uint64_t file_size = 1024*1024;
    uint64_t h_num_file = 1024;
    uint64_t data_size = h_num_file*file_size;

    // inicjalizacja i alokacja pamięci na hoście
	uint8_t *h_data = (uint8_t*)malloc(data_size);
	double *h_tab_Pvalue = (double *)malloc(sizeof(double)*h_num_file);

    // generowanie plików
    for(int i=0; i<data_size; i++){
        h_data[i]=rand()%256;
    }
    cout << "Wygenerowano dane o rozmiarze " << data_size << "\n";

    int threadsPerBlock = 128;
    int blocksPerGrid = (h_num_file+threadsPerBlock-1)/threadsPerBlock;;

    ////////////////////// GPU ////////////////////////////////
    auto GPU_t_start = std::chrono::high_resolution_clock::now();

    uint8_t * d_data;
    double *d_tab_Pvalue;
    cudaMalloc((void **)&d_data, data_size); 
    cudaMalloc((void **)&d_tab_Pvalue, sizeof(double)*h_num_file); 
    cudaMemcpy(d_data,h_data,data_size, cudaMemcpyHostToDevice);


    cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, h_num_file, file_size, d_tab_Pvalue);

    cout << "GPU zakończyło obliczenia\n";

    cudaMemcpy(h_tab_Pvalue, d_tab_Pvalue, sizeof(double)*h_num_file,cudaMemcpyDeviceToHost);
    auto GPU_t_end = std::chrono::high_resolution_clock::now();
    double GPU_time = std::chrono::duration<double, std::milli>(GPU_t_end-GPU_t_start).count();

    ////////////////////////// CPU //////////////////////
    auto CPU_t_start = std::chrono::high_resolution_clock::now();
    double * CPU_P_value = (double *)malloc(sizeof(double)*h_num_file);
    for(int i=0;i<h_num_file;i++){
        //CPU_P_value[i] = CuSum(&h_data[file_size*i],file_size);
    }
    auto CPU_t_end = std::chrono::high_resolution_clock::now();
    double CPU_time = std::chrono::duration<double, std::milli>(CPU_t_end-CPU_t_start).count();

    cout << "CPU zakończyło obliczenia\n";


    ////////////////////// CHECK RESOULT /////////////
    // for(int i=0;i<h_num_file;i++){
    //     if (abs(h_tab_Pvalue[i] - CPU_P_value[i]) > 1e-5) {
    //         cout << h_tab_Pvalue[i] << " " << CPU_P_value[i] << "\n";
    //         fprintf(stderr, "Result verification failed at index %i!\n",i);
    //         exit(EXIT_FAILURE);
    //     }
    // }

    cout<<"Czas wykonania GPU: "<<GPU_time<< "\n";
    cout<<"Czas wykonania CPU: "<<CPU_time<< "\n";

    free(h_tab_Pvalue);
    free(h_data);

    cudaFree(d_tab_Pvalue);
    cudaFree(d_data);

}


// test wydajności na wileu plikach, pliki generowanie 
void test_wydajnosci_na_plikach(){
    uint64_t start_file_size = 1024*512;
    uint64_t max_file_size = 1024*512;
    uint64_t max_file_num = 1024*20;
    cout<<"Alokacja pamięci na CPU\n";
    uint64_t max_data_size = max_file_size*max_file_num;
    cout<< max_data_size <<endl;
    uint8_t *h_data = (uint8_t*)malloc(max_data_size);
    cout << "Generowanie danych\n";
    // generowanie plików
    for(int i=0; i<max_data_size; i++){
        h_data[i]=rand()%256;
    }
    cout << "Alokacja pamięcia na GPU\n";
    uint8_t * d_data;
        cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "3 Failed to launch cuda_CuSum kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void **)&d_data, max_data_size); 
    cout << "Kopiowanie danych do GPU\n";
    cudaMemcpy(d_data,h_data, max_data_size, cudaMemcpyHostToDevice);

    ofstream file;
    file.open("wynik.txt");

    file << "File size\tFile num\ttime\n";
    
    

    for(uint64_t file_size = start_file_size;file_size<=max_file_size;file_size+=32){
        cout << file_size << "\n";
        for(uint64_t h_num_file = 16;h_num_file<max_file_num; h_num_file+=16){
            // cout<<h_num_file<<"\n";
            //uint64_t h_num_file = num_file_tab[file_i];
            //uint64_t data_size = h_num_file*file_size;

            // inicjalizacja i alokacja pamięci na hoście
            double *h_tab_Pvalue = (double *)malloc(sizeof(double)*h_num_file);

            ////////////////////// GPU ////////////////////////////////
            double *d_tab_Pvalue;
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "1 Failed to launch cuda_CuSum kernel (error code %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaMalloc((void **)&d_tab_Pvalue, sizeof(double)*h_num_file); 
            int threadsPerBlock = 32;
            int blocksPerGrid = (h_num_file+threadsPerBlock-1)/threadsPerBlock;;
        err = cudaSuccess;
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "2 Failed to launch cuda_CuSum kernel (error code %s)!\n",
					cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            cuda_CuSum<<<blocksPerGrid,threadsPerBlock>>>(d_data, h_num_file, file_size, d_tab_Pvalue);
            
        err = cudaSuccess;
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			fprintf(stderr, "Failed to launch cuda_CuSum kernel (error code %s)!\n",
					cudaGetErrorString(err));
            cout<<"blocksPerGrid "<<blocksPerGrid<<" threadsPerBlock "<<threadsPerBlock<<endl;
            cout<<"h_num_file "<<h_num_file<<" file_size "<<file_size<<endl;
			exit(EXIT_FAILURE);
		}

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float GPU_time = 0;
            cudaEventElapsedTime(&GPU_time, start, stop);

            cudaMemcpy(h_tab_Pvalue, d_tab_Pvalue, sizeof(double)*h_num_file,cudaMemcpyDeviceToHost);
            //file<< file_size <<"\t"<<h_num_file<<"\t"<<GPU_time <<"\n";
            file <<h_num_file<<"\t" << GPU_time << "\n";

            free(h_tab_Pvalue);
            cudaFree(d_tab_Pvalue);

        }

    }
    cudaFree(d_data);
    free(h_data);
    //uint64_t file_size = 1024;//*1024*4;
    
}


int main(int argc, char *argv[]){

    Test();
    one_block();
    

    return 0;
}
