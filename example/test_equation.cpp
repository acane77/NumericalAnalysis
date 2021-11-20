#include <iostream>
#include "equation.h"

using namespace std;

float f(float x) { return 666 -  6 * x ; }

int main() {
    try {
        cout << "Solve equation: Ax=b, with:\n";
        matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
        matrix_t b = { 20,33,12 };
        PRINT_MAT(A);
        PRINT_MAT(b);
        matrix_t x0 = b.zerosLike();
        PRINT_MAT(x0);

        matrix_t x;
        cout << "Using Jacobi Iteration: " << endl;
        jacobiInteration(A, b, 1e-6f, x);
        PRINT_MAT(x);

        cout << "Using Gauss-Seidel Iteration: " << endl;
        gaussSeidelIteration(A, b, 1e-6f, x);
        PRINT_MAT(x);

        cout << "Using SOR Iteration: " << endl;
        SORIteration(A, b, 1.0f, 1e-6f, x);
        PRINT_MAT(x);

        cout << "Using Gaussian Iteration: " << endl;
        gaussianReduce(A, b, x);
        PRINT_MAT(x);
        
        cout << "Solve non-linear equation y=f(x): \n";
        float _x;
        cout << "Using Newton Iteration: " << endl;
        int ret = newtonIteration(f, -1.0f, 1.0f, 1e-6f, _x);
        PRINT_MAT(_x);
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
