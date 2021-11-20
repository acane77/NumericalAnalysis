#ifndef LIBNUMANALYSIS_EQUATION_H
#define LIBNUMANALYSIS_EQUATION_H

#include "matrix.h"

#define MAX_ITERATION_COUNT 300

#define PRINT_MAT(M) std::cout << #M << " = " << (M)

#define MATRIX_API template <class ElementTy>
#define MATRIX_T   Matrix<ElementTy>
#define MATRIX_FROM_VIEW(view_name, mat) view_name<ElementTy>(mat).clone()
#define ELEMENT_T  ElementTy

enum iter_result_t {
    ITER_OK = 0,
    ITER_UNCOVERAGED = 1,
    ITER_REQUIRE_SQUARE_MATRIX = 2
};

MATRIX_API
iter_result_t jacobiInteration(MATRIX_T A, MATRIX_T b, MATRIX_T x0, ELEMENT_T tol, MATRIX_T& out_result) {
    if (b.getRowCount() == 1)
        b = b.transpose();
    if (x0.getRowCount() == 1)
        x0 = x0.transpose();
    MATRIX_T x = x0;
    // A = D-l-U
    MATRIX_T L = -(A.lowerTriangle()), U = -(A.upperTriangle()), D = A.diagonal();
    // B_J = D^-1*(L+U)
    MATRIX_T D_r = MATRIX_FROM_VIEW(ElementWiseReciprocalView, D);
    MATRIX_T B_J = D_r * (L + U);
    // f = D^-1*b
    MATRIX_T f = D_r * b;
    ElementTy drt_x_l2norm;
    int iter_count = 0;
    do {
        iter_count++;
        // x_1 = B_J * x_0 + f
        MATRIX_T x1 = B_J * x + f;
        drt_x_l2norm = (x - x1).l2Norm();
        x = x1;
    } while (drt_x_l2norm >= tol && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return ITER_UNCOVERAGED;
    out_result = x;
    return ITER_OK;
}

#endif //LIBNUMANALYSIS_EQUATION_H
