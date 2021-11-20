#include <iostream>
#include "equation.h"

using namespace std;

#define ElementTy float

int main() {
    try {
        matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
        matrix_t b = { 20,33,12 };
        matrix_t x0 = b.zerosLike();
        matrix_t x;
        gaussSeidelIteration(A, b, 1e-6f, x);
        PRINT_MAT(x);
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
