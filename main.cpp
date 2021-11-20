#include <iostream>
#include "equation.h"

using namespace std;

#define ElementTy float

int main() {
    try {
        matrix_t A = {{2,0,0},{0,3,0},{0,0,4}};
        matrix_t b = {20,33,12};
        matrix_t x;
        int n = A.getColumnCount();
        //jacobiInteration(A, b, 1e-6f, x);
        matrix_t r = A.inverse();
        PRINT_MAT(A);
        PRINT_MAT(A.inverse());

        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
