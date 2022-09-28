#ifndef IO_HPP
#define IO_HPP

#include <vector>
#include <string>

// Zaczytanie pliku do wektora
std::vector<uint8_t> read_file(std::string path);

// Funkcja zwraca wektora z nazwami plików z katalogu dir_path o rozszerzeniu extension
std::vector<std::string> file_names(std::string dir_path, std::string extension);

// Funkcja zwraca wektor z wektorami zawierającymi dane z plików, których nazwy znajdują się w file_names
std::vector<std::vector<uint8_t>> read_all_files(std::vector<std::string> file_names);

// Wypisanie danych w postaci binarnej na standardowe wyjście
void print_data(std::vector<uint8_t> data, int m);
// Wypisanie danych w postaci binarnej na standardowe wyjście
void print_data(uint8_t *data, int m);

// Funkcja zwraca liczbę 64 bitową, której najstarze bity to bity znajdujące się od pozycji stard to pozycji end w wektorze data
uint64_t extract_bits(std::vector<uint8_t> data, int start, int len);
// Funkcja zwraca liczbę 64 bitową, której najstarze bity to bity znajdujące się od pozycji stard to pozycji end w tablicy data
uint64_t extract_bits(uint8_t *data, int start, int len);

#endif