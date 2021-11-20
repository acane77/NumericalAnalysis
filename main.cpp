#include <iostream>
#include "equation.h"

using namespace std;

float f(float x) { return 666 -  6 * x; }

int main() {
    try {
        matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
        matrix_t b = { 20,33,12 };
        matrix_t x0 = b.zerosLike();
        matrix_t x;
        jacobiInteration(A, b, 1e-6f, x);
        PRINT_MAT(x);
        gaussSeidelIteration(A, b, 1e-6f, x);
        PRINT_MAT(x);
        SORIteration(A, b, 1.0f, 1e-6f, x);
        PRINT_MAT(x);
        float _x;
        int ret = newtonIteration(f, -1.0f, 1.0f, 1e-6f, _x);
        PRINT_MAT(_x);
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
