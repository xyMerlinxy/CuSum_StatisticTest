#include <vector>

#include "./../io.hpp"
#include "./../CuSum.hpp"

using namespace std;


bool Test_1_CuSum_cpp_v1(){
    double file_P_value = 0.451231;
    vector<uint8_t> file = read_file("test/test_data/data.sha1");
    double P_value = CuSum(file.data(), file.size());
    return abs(file_P_value-P_value)<1e-5;
}
bool Test_1_CuSum_cpp_v2(){
    double file_P_value = 0.547944;
    vector<uint8_t> file = read_file("test/test_data/data.bad_rng");
    double P_value = CuSum(file.data(), file.size());
    return abs(file_P_value-P_value)<1e-5;
}
bool Test_1_CuSum_cpp_v3(){
    double file_P_value = 0.405915;
    vector<uint8_t> file = read_file("test/test_data/100_1_0.bin");
    double P_value = CuSum(file.data(), file.size());
    return abs(file_P_value-P_value)<1e-5;
}
bool Test_1_CuSum_cpp_v4(){
    double file_P_value = 0.434313;
    vector<uint8_t> file = read_file("test/test_data/1000000_1_0.bin");
    double P_value = CuSum(file.data(), file.size());
    return abs(file_P_value-P_value)<1e-5;
}

int Test_1(){
    if (Test_1_CuSum_cpp_v1()){
        cout << "   Test CuSum.cpp_1_1\tOK\n";
    }
    else{
        cout << "   Test CuSum.cpp_1_1\tFAILED\n";
        return false;
    }

    if (Test_1_CuSum_cpp_v2()){
        cout << "   Test CuSum.cpp_1_2\tOK\n";
    }
    else{
        cout << "   Test CuSum.cpp_1_2\tFAILED\n";
        return false;
    }

    if (Test_1_CuSum_cpp_v3()){
        cout << "   Test CuSum.cpp_1_3\tOK\n";
    }
    else{
        cout << "   Test CuSum.cpp_1_3\tFAILED\n";
        return false;
    }

    if (Test_1_CuSum_cpp_v4()){
        cout << "   Test CuSum.cpp_1_4\tOK\n";
    }
    else{
        cout << "   Test CuSum.cpp_1_4\tFAILED\n";
        return false;
    }

    return true;
}