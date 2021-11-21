#include <iostream>
#include "test.h"
#include "equation.h"

using namespace std;

float f(float x) { return 666 -  6 * x ; }

TEST(equation, test_jacobi) {
    cout << "Solve equation: Ax=b, with:\n";
    matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
    matrix_t b = { 20,33,12 };
    PRINT_MAT(A);
    PRINT_MAT(b);

    matrix_t x;
    cout << "Using Jacobi Iteration: " << endl;
    jacobiInteration(A, b, 1e-6f, x);
    PRINT_MAT(x);
}

TEST(equation, test_gauss_seide) {
    matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
    matrix_t b = { 20,33,12 };
    matrix_t x;

    cout << "Using Gauss-Seidel Iteration: " << endl;
    gaussSeidelIteration(A, b, 1e-6f, x);
    PRINT_MAT(x);
}

TEST(equation, test_sor) {
    matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
    matrix_t b = { 20,33,12 };
    matrix_t x;

    cout << "Using SOR Iteration: " << endl;
    SORIteration(A, b, 1.0f, 1e-6f, x);
    PRINT_MAT(x);
}

TEST(equation, test_gauss) {
    matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
    matrix_t b = { 20,33,12 };
    matrix_t x;

    cout << "Using Gaussian Iteration: " << endl;
    gaussianReduce(A, b, x);
    PRINT_MAT(x);
}


TEST(equation, test_gauss_max_col_pivot) {
    matrix_t A = { {0.000100f, 1.00f} , {1.00f, 1.00f} };
    matrix_t b = { 1.00f, 2.00f };
    PRINT_MAT(A);
    PRINT_MAT(b);
    matrix_t x;

    cout << "Using Max Column Pivot Gaussian Iteration: " << endl;;
    gaussianReduceWithMaximalColumnPivot(A, b, x);
    PRINT_MAT(x);
    float x0 = x[0][0], x1 = x[1][0];
    ASSERT_LE(x0, 1.5f);
    ASSERT_GT(x1, 0.5f);
}

TEST_MAIN()