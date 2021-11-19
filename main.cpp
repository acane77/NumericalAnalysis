#include <iostream>
#include "matrix.h"

using namespace std;

int main() {
    try {
        matrix_t m1 = {1, 2, 3, 4};
        matrix_t m2 = {{1, 4, 6, 7}};
        m1 = m1.transpose();
        matrix_t m3 = m1 + m2;
        m3.print();
        cout << endl;
        row_view_t r33 = m3.getRow(1);
        column_view_t c33 = m3.getColumn(1);
        mutable_column_view_t r33_1 = m3.getMutableColumn(1);
        r33_1.set(2, 123);
        m3[2][2] = 322;
        auto t = m3[2];
        t[3] = 555;
        m3.print();
        //MatrixReshapeView<float>(m1, 2, 2).print();
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
