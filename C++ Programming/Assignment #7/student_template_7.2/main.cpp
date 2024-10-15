#include "submission/complex_number.h"
#include <cstdlib>
#include <iostream>
#include <sstream>

int main(){
    ComplexNumber<double> cn1(1,1);
    ComplexNumber<double> cn2(2,3);
    ComplexNumber<double> cn3 = cn1 + cn2;
    std::cout << cn3 << std::endl;

    std::stringstream out_stream;
    out_stream << cn1 << std::endl;
    std::cout << out_stream.str() << std::endl;

    return EXIT_SUCCESS;
}