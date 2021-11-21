#include "test.h"
#include "matrix.h"

TEST(matrix, test_argmax) {
	matrix_t mat = { { 1,2,3,4 }, {2,3,7,3} };
	PRINT_MAT(mat);
	float x;
	x = mat.argmax();
	PRINT_MAT(mat.argmax());
	ASSERT_EQ(x, 6);

	PRINT_MAT(mat.argmax(0));
	PRINT_MAT(mat.argmax(1));

	ASSERT_EQ(1, mat.getColumn(2).argmax());
	ASSERT_EQ(0, mat.getColumn(3).argmax());
	ASSERT_EQ(2, mat[1].argmax());
}

TEST(matrix, test_swap_rows) {
	matrix_t mat = matrix_t::randn(5, 5);
	mat[1][1] = 5; mat[2][1] = 10;
	PRINT_MAT(mat);
	mat.swapRow(1, 2);
	PRINT_MAT(mat);
	ASSERT_EQ(10, mat[1][1]);
	ASSERT_EQ(5, mat[2][1]);

	mat = matrix_t::randn(5, 1);
	PRINT_MAT(mat);
	mat.swapRow(1, 2);
	PRINT_MAT(mat);
}

TEST(matrix, test_swap_col) {
	matrix_t mat = matrix_t::randn(5, 5);
	mat[1][1] = 5; mat[1][2] = 10;
	PRINT_MAT(mat);
	mat.swapCol(1, 2);
	PRINT_MAT(mat);
	ASSERT_EQ(10, mat[1][1]);
	ASSERT_EQ(5, mat[1][2]);
}

TEST(matrix, test_slice) {
	matrix_t mat = matrix_t::randn(6, 6);
	PRINT_MAT(mat);
	auto sliced = mat.slice(2, 2, -1, -1);
	PRINT_MAT(sliced);
	auto sliced2 = sliced.slice(2, 0, 2, 3);
	PRINT_MAT(sliced2);
	auto sliced3 = sliced2.slice1d(1, 2);
	PRINT_MAT(sliced3);
	PRINT_MAT(sliced3.argmax());
}

TEST_MAIN()