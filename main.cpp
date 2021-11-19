#include <iostream>
#include "matrix.h"

using namespace std;

int main() {
    try {
        matrix_t mat = matrix_t::randn(3,4);
        mat.print("\t");
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
