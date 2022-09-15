#ifndef IO_H
#define IO_H


#include "io.cpp"

std::vector<uint8_t> read_file(std::string path);
std::vector<std::string> file_names(std::string dir_path, string extension);
std::vector<std::vector<uint8_t>> read_all_files(std::vector<std::string> file_names);

void print_data(std::vector<uint8_t> data, int m);
void print_data(uint8_t *data, int m);

uint64_t extract_bits(std::vector<uint8_t> data, int start, int len);
uint64_t extract_bits(uint8_t *data, int start, int len);

#endif