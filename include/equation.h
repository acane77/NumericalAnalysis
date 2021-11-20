#ifndef LIBNUMANALYSIS_EQUATION_H
#define LIBNUMANALYSIS_EQUATION_H

#include "matrix.h"
#include "common_def.h"

// =====================  LINEAR EQUATIONS  =====================

/// Perform Jacobi iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return NA_OK if successful
MATRIX_API
na_result_t jacobiInteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result);

/// Perform Gauss-Seidel iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return NA_OK if successful
MATRIX_API
na_result_t gaussSeidelIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result);

/// Perform SOR iteration on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param w           [IN]  omega (0 < omega < 2)
/// \param tol         [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return NA_OK if successful
MATRIX_API
na_result_t SORIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T w, ELEMENT_T tol, MATRIX_T& out_result);

/// Perform Gaussian reduce on linear equation Ax=b
/// \param A           [IN]  coefficient matrix A
/// \param b           [IN]  b
/// \param out_result  [OUT] result x
/// \return a value indicates if reduction is successful, return NA_OK if successful
MATRIX_API
na_result_t gaussianReduce(MATRIX_T _A, MATRIX_T _b, MATRIX_T& out_result);

// ==================  NON-LINEAR EQUATIONS  ===================

/// Perform simple Newton iteration on non-linear equation f(x)=0
/// \param f           [IN]  single-meta function f(x)
/// \param x0          [IN]  initial value x_0
/// \param x1          [IN]  initial value x_1
/// \param epsilon     [IN]  order of convergence
/// \param out_result  [OUT] result x
/// \return a value indicates if iteration is successful, return NA_OK if successful
EQU_API
na_result_t newtonIteration(SINGLE_META_FUNCTION_T f, VALUE_T x0, VALUE_T x1, VALUE_T epsilon, VALUE_T& out_result);

// ==============================================================
//                       IMPLEMENTATIONS
// ==============================================================

#define PERFORM_SQUARE_MATRIX_CHECK(mat) do { \
    if ((mat).getColumnCount() != (mat).getRowCount())\
        return NA_REQUIRE_SQUARE_MATRIX; \
} while (0)

#define TRANSPOSE_INPUT_TO_HERIZONTAL(mat) \
    do {\
        if ((mat).getRowCount() == 1)\
            (mat) = (mat).transpose();\
    } while (0)

#define PERFORMER_MATRIX_COLUMN_CHECK(mat, expected_row_count) \
    do {\
        if ((mat).getColumnCount() != (expected_row_count))\
            return NA_INVADE_ARG;\
    } while (0)

#define ITER_CHECKARGS() \
    do {            \
    PERFORM_SQUARE_MATRIX_CHECK(A);\
    TRANSPOSE_INPUT_TO_HERIZONTAL(b);\
    PERFORMER_MATRIX_COLUMN_CHECK(b, 1);\
    } while(0)

MATRIX_API
na_result_t jacobiInteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result) {
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
    ELEMENT_T drt_x_l2norm;
    int iter_count = 0;
    do {
        iter_count++;
        // x_1 = B_J * x_0 + f
        MATRIX_T x1 = B_J * x + f;
        drt_x_l2norm = (x - x1).l2Norm();
        x = x1;
    } while (drt_x_l2norm >= tol && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return NA_ITER_UNCOVERAGED;
    out_result = x;
    return NA_OK;
}

MATRIX_API
na_result_t gaussSeidelIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T tol, MATRIX_T& out_result) {
    ITER_CHECKARGS();
    MATRIX_T x0 = b.zerosLike();
    MATRIX_T x = x0;
    // A = D+l+U
    MATRIX_T L = A.lowerTriangle(), U = A.upperTriangle(), D = A.diagonal();
    MATRIX_T DpL_inv = (D + L).inverse();  // (D+L)^-1
    // f = (D+L)^-1b
    MATRIX_T f = DpL_inv * b;
    MATRIX_T B_G = -(DpL_inv * U);
    ELEMENT_T drt_x_l2norm;
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
        return NA_ITER_UNCOVERAGED;
    out_result = x;
    return NA_OK;
}

MATRIX_API
na_result_t SORIteration(MATRIX_T A, MATRIX_T b, ELEMENT_T w, ELEMENT_T tol, MATRIX_T& out_result) {
    ITER_CHECKARGS();
    if (w <= 0 || w >= 2)
        return NA_INVADE_ARG;
    MATRIX_T x0 = b.zerosLike();
    MATRIX_T x = x0;
    // A = D+l+U
    MATRIX_T L = -(A.lowerTriangle()), U = -(A.upperTriangle()), D = A.diagonal();
    MATRIX_T DmwL_inv = (D - L * w).inverse();  // (D-L)^-1
    // f = w(D-wL)^-1b
    MATRIX_T f = DmwL_inv * w * b;
    // B_w = (D-wL)^-1((1-w)D+wU)
    MATRIX_T B_w = DmwL_inv * (D * (1-w) + U * w);
    ELEMENT_T drt_x_l2norm;
    int iter_count = 0;
    do {
        iter_count++;
        // x_1 = B_w * x_0 + f
        MATRIX_T x1 = B_w * x + f;
        drt_x_l2norm = (x - x1).l2Norm();
        x = x1;
    } while (drt_x_l2norm >= tol && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return NA_ITER_UNCOVERAGED;
    out_result = x;
    return NA_OK;
}

MATRIX_API
na_result_t gaussianReduce(MATRIX_T _A, MATRIX_T _b, MATRIX_T& out_result) {
    MATRIX_T b = _b;
    MATRIX_T A = _A.clone();
    ITER_CHECKARGS();
    // (A|b) = (E|x)
    int n = _A.getColumnCount();
    MATRIX_T x = b.clone();

    for (int i = 0; i < n; i++) {
        ELEMENT_T A_ii = A[i][i];
        A[i] = A[i] / A_ii; x[i][0] = x[i][0] / A_ii;
        for (int j = i + 1; j < n; j++) {
            ELEMENT_T A_ji = A[j][i];
            A[j] = A[j].clone() - A[i] * A_ji; x[j][0] = x[j][0] - x[i][0] * A_ji;
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            ELEMENT_T A_ji = A[j][i];
            A[j] = A[j].clone() - A[i] * A_ji; x[j][0] = x[j][0] - x[i][0] * A_ji;
        }
    }
    out_result = x;
    return NA_OK;
}

EQU_API 
na_result_t newtonIteration(SINGLE_META_FUNCTION_T f, VALUE_T x0, VALUE_T x1, VALUE_T epsilon, VALUE_T& out_result) {
    VALUE_T drt;
    int iter_count = 0;
    VALUE_T x;
    do {
        x = x1 - f(x1) / (f(x1) - f(x0)) * (x1 - x0);
        drt = x - x1;
        x0 = x1;
        x1 = x;
    } while (drt >= epsilon && iter_count < MAX_ITERATION_COUNT);
    if (iter_count >= MAX_ITERATION_COUNT)
        return NA_ITER_UNCOVERAGED;
    out_result = x;
    return NA_OK;
}

// undefine some references
#undef PERFORM_SQUARE_MATRIX_CHECK
#undef TRANSPOSE_INPUT_TO_HERIZONTAL
#undef PERFORMER_MATRIX_COLUMN_CHECK
#undef ITER_CHECKARGS

#endif //LIBNUMANALYSIS_EQUATION_H
