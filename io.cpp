#include <fstream>
#include <iostream>
#include <filesystem>

#include "io.hpp"

using namespace std;

namespace fs = filesystem;

vector<uint8_t> read_file(string path){
    if(!fs::is_regular_file(path)){
        cout << "\"" << path << "\" nie jest ścieżką do pliku" << endl;
        exit(EXIT_FAILURE);
    }

    ifstream file;
    file.open(path, ios::binary|ios::in);
    size_t len = fs::file_size(path);

    vector<uint8_t> data;
    data.resize(len);
    file.read((char*)data.data(), len);
    file.close();
    return data;
}

vector<string> file_names(string dir_path, string extension){
    if(!fs::is_directory(dir_path)){
        cout << "\"" << dir_path << "\" nie jest ścieżką do katalogu" << endl;
        exit(EXIT_FAILURE);
    }

    vector<string> names;
    for (const auto & entry : fs::directory_iterator(dir_path)){
        if((string)entry.path().extension() == extension){
            names.push_back((string)entry.path());
        }
    }
    return names;
}

vector<vector<uint8_t>> read_all_files(vector<string> file_names){
    vector<vector<uint8_t>> files_data;
    
    for(auto path: file_names){
        files_data.push_back(read_file(path));
    }
    return files_data;
}

void print_data(uint8_t *data, int m){
    for(uint64_t index = 0; index < m/8; index++){
        uint8_t byte = data[index];
        for(int i=0; i<8; i++){
            if(byte & 0x80) cout << "1";
            else cout << "0";
            byte <<= 1;
        }
    }
    cout << endl;
}

void print_data(vector<uint8_t> data, int m){
    print_data(data.data(), m);
}
 
uint64_t extract_bits(uint8_t *data, int start, int len){
    uint64_t index = start / 8;

    uint64_t value = 0;
    int empty_bits = 64;

    value += ((uint64_t)data[index]) << (empty_bits - 8 + start%8);
    empty_bits -= 8 - start%8;
    len -= 8 - start%8;
    index++;

    for(; len>=8; len-=8){
        value += ((uint64_t)data[index]) << empty_bits - 8;
        empty_bits -= 8;
        index++;
    }

    value += ((uint64_t)(data[index] >> (8-len))) << (empty_bits-len);
    return value;
}

uint64_t extract_bits(vector<uint8_t> data, int start, int len){
    return extract_bits(data.data(), start, len);
}
