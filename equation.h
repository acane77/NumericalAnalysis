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
    ITER_OK = 0,                      // iteration success
    ITER_UNCOVERAGED = 1,             // equation is uncoveraged
    ITER_REQUIRE_SQUARE_MATRIX = 2,   // require a square matrix for input
    ITER_INVADE_ARG = 3,              // invalid argument
};

/// Perform Jacobi iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return ITER_OK if successful
MATRIX_API
iter_result_t jacobiInteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result);

/// Perform Gauss-Seidel iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return ITER_OK if successful
MATRIX_API
iter_result_t gaussSeidelIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result);

/// Perform SOR iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param w           [IN]  omega (0 < omega < 2)
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return ITER_OK if successful
MATRIX_API
iter_result_t SORIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T w, ELEMENT_T tol, MATRIX_T& out_result);

// ==========================================================

#define PERFORM_SQUARE_MATRIX_CHECK(mat) do { \
    if ((mat).getColumnCount() != (mat).getRowCount())\
        return ITER_REQUIRE_SQUARE_MATRIX; \
} while (0)

#define TRANSPOSE_INPUT_TO_HERIZONTAL(mat) \
    do {\
        if ((mat).getRowCount() == 1)\
            (mat) = (mat).transpose();\
    } while (0)

#define PERFORMER_MATRIX_COLUMN_CHECK(mat, expected_row_count) \
    do {\
        if ((mat).getColumnCount() != (expected_row_count))\
            return ITER_INVADE_ARG;\
    } while (0)

#define ITER_CHECKARGS() \
    do {            \
    PERFORM_SQUARE_MATRIX_CHECK(A);\
    TRANSPOSE_INPUT_TO_HERIZONTAL(b);\
    PERFORMER_MATRIX_COLUMN_CHECK(b, 1);\
    } while(0)

MATRIX_API
iter_result_t jacobiInteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result) {
    ITER_CHECKARGS();
    MATRIX_T x0 = b.zerosLike();
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

MATRIX_API
iter_result_t gaussSeidelIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result) {
    ITER_CHECKARGS();
    MATRIX_T x0 = b.zerosLike();
    MATRIX_T x = x0;
    // A = D+l+U
    MATRIX_T L = A.lowerTriangle(), U = A.upperTriangle(), D = A.diagonal();
    MATRIX_T DpL_inv = (D + L).inverse();  // (D+L)^-1
    // f = (D+L)^-1b
    MATRIX_T f = DpL_inv * b;
    MATRIX_T B_G = -(DpL_inv * U);
    ElementTy drt_x_l2norm;
    int iter_count = 0;
    do {
        iter_count++;
        // x_1 = -(D+L)^-1Ux_0+(D+L)^-1b
        // x_1 = B_G * x_0 + f
        MATRIX_T x1 = B_G * x + f;
        drt_x_l2norm = (x - x1).l2Norm();
        x = x1;
    } while (drt_x_l2norm >= tol && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return ITER_UNCOVERAGED;
    out_result = x;
    return ITER_OK;
}

MATRIX_API
iter_result_t SORIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T w, ELEMENT_T tol, MATRIX_T& out_result) {
    ITER_CHECKARGS();
    if (w <= 0 || w >= 2)
        return ITER_INVADE_ARG;
    MATRIX_T x0 = b.zerosLike();
    MATRIX_T x = x0;
    // A = D+l+U
    MATRIX_T L = -(A.lowerTriangle()), U = -(A.upperTriangle()), D = A.diagonal();
    MATRIX_T DmwL_inv = (D - L * w).inverse();  // (D-L)^-1
    // f = w(D-wL)^-1b
    MATRIX_T f = DmwL_inv * w * b;
    // B_w = (D-wL)^-1((1-w)D+wU)
    MATRIX_T B_w = DmwL_inv * (D * (1-w) + U * w);
    ElementTy drt_x_l2norm;
    int iter_count = 0;
    do {
        iter_count++;
        // x_1 = B_w * x_0 + f
        MATRIX_T x1 = B_w * x + f;
        drt_x_l2norm = (x - x1).l2Norm();
        x = x1;
    } while (drt_x_l2norm >= tol && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return ITER_UNCOVERAGED;
    out_result = x;
    return ITER_OK;
}

// undefine some references
#undef PERFORM_SQUARE_MATRIX_CHECK
#undef TRANSPOSE_INPUT_TO_HERIZONTAL
#undef PERFORMER_MATRIX_COLUMN_CHECK
#undef ITER_CHECKARGS

#endif //LIBNUMANALYSIS_EQUATION_H
