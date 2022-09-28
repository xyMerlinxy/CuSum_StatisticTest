#include "Test.hpp"

#include "Test_x.hpp"

#include<iostream>


bool Test(){
    std::cout << "Testing:" <<std::endl;

    bool resoult = Test_1()&Test_2();
    if (resoult) std::cout << "All test passed\n";
    else std::cout << "Something goes wrong\n";
    
    return resoult;
}