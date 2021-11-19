#ifndef LIBNUMANALYSIS_MATRIX_H
#define LIBNUMANALYSIS_MATRIX_H

#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>
#include <exception>
#include <vector>
#include <functional>
#include <algorithm>

#define PERIOR_DEFINE_CLASS(class_name) template <class T> class class_name

PERIOR_DEFINE_CLASS(Matrix);
PERIOR_DEFINE_CLASS(ColumnView);
PERIOR_DEFINE_CLASS(RowView);
PERIOR_DEFINE_CLASS(MutableRowView);
PERIOR_DEFINE_CLASS(MutableColumnView);

template <class ElementTy=float>
class MatrixBase {
protected:
    // Represents the real columns and rows of matrix i memory,
    // Note: even in matrix views, this numCol and numRow is also represent
    //       the real data size in memory.
    int numCol = 0;
    int numRow = 0;
    std::shared_ptr<ElementTy> data = nullptr;

    MatrixBase(int numCol, int numRow, ElementTy *data) : numCol(numCol), numRow(numRow), data(data) {}
    MatrixBase() = default;

    ElementTy max(ElementTy a, ElementTy b) const { return a > b ? a : b; }
    ElementTy min(ElementTy a, ElementTy b) const { return a < b ? a : b; }

public:
    MatrixBase(const MatrixBase& another);
    // Get element at given row and column
    virtual ElementTy get(int row, int col) { return *(data.get() + (row * numCol) + col); }
    // Print the matrix
    void print(const char* dstr = nullptr);
    // Perform broadcast add
    Matrix<ElementTy> operator+ (Matrix<ElementTy>& another) ;
    // Perform broadcast substract
    Matrix<ElementTy> operator- (Matrix<ElementTy>& another) ;
    // Perform broadcast multiply
    Matrix<ElementTy> operator* (Matrix<ElementTy>& another) ;
    // Perform broadcast dot multiply
    Matrix<ElementTy> dot(Matrix<ElementTy>& another) ;
    // Clone into another mutable matrix
    Matrix<ElementTy> clone();
    // Get logical column count
    virtual int getColumnCount() { return numCol; }
    // Get logical row count
    virtual int getRowCount() { return numRow; }
    // Get count of elements
    virtual int getElemCount() { return this->getColumnCount() * this->getRowCount(); }
    // Get size of elements
    virtual size_t size() { return this->getElemCount(); }
    // Get the transposed matrix
    Matrix<ElementTy> transpose();
    // Resize matrix to new size
    Matrix<ElementTy> resize(int r, int c);
    // Reshape matrix to new size
    Matrix<ElementTy> reshape(int r, int c);
    // boardcast to new size (will not create new object)
    MatrixBase<ElementTy> boardcastTo(int r, int c);
    // boardcast binary op
    Matrix<ElementTy> boardcastBinaryOperator(Matrix<ElementTy>& another, const std::function<ElementTy (ElementTy, ElementTy)>& op_handler);
    // get a row
    RowView<ElementTy> getRow(int row);
    // get a row
    RowView<ElementTy> operator[] (int row);
    // get a column
    ColumnView<ElementTy> getColumn(int col);
    // get raw data in memory
    virtual ElementTy* rawData() { return data.get(); }

    virtual ~MatrixBase() = default;

private:
    ElementTy* getRawData(int row, int col) { return (data.get() + (row * numCol) + col); }
};

template <class ElementTy=float>
class MutableMatrixBase : public MatrixBase<ElementTy> {
public:
    virtual void set(int row, int col, ElementTy value) = 0;
};

template <class ElementTy=float>
class Matrix : public MutableMatrixBase<ElementTy> {
public:
    Matrix(const std::initializer_list<std::initializer_list<ElementTy>>& data);
    Matrix(const std::initializer_list<ElementTy>& data);
    Matrix(ElementTy number);
    Matrix(int nr, int nc);
    // set value of an element
    void set(int row, int col, ElementTy value) override;
    // swap two rows
    void swapRow(int row1, int row2);
    // swap two columns
    void swapCol(int col1, int col2);
    // swap two elements
    void swapElem(int col1, int row1, int col2, int row2);
    // get all-zero matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
    zeros(int r, int c);
    // get all-one matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
    ones(int r, int c);
    // get identity matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
    identity(int size, ElementTy value=1);
    // get a row
    MutableRowView<ElementTy> getMutableRow(int row);
    // get a row
    MutableRowView<ElementTy> operator[] (int row) { return std::move(getMutableRow(row)); }
    // get a column
    MutableColumnView<ElementTy> getMutableColumn(int col);

    ~Matrix();
};

template <class ElementTy=float>
class VectorView : public MatrixBase<ElementTy> {
public:
    VectorView(MatrixBase<ElementTy> matrix) : MatrixBase<ElementTy>(matrix) { }
    virtual ElementTy get(int index) = 0;
};

template <class ElementTy=float>
class MutableVectorView : public MatrixBase<ElementTy> {
protected:
    void set_elem(int row, int col, ElementTy value) {
        *(MatrixBase<ElementTy>::data.get() + (row * MatrixBase<ElementTy>::numCol) + col) = value;
    }
public:
    MutableVectorView(MatrixBase<ElementTy> matrix) : MatrixBase<ElementTy>(matrix) { }
    virtual ElementTy get(int index) = 0;
    virtual void set(int index, ElementTy value) = 0;
};

template <class ElementTy=float>
class MatrixBoardcastView : public MatrixBase<ElementTy> {
    float scaleRow, scaleCol;
    int newCol, newRow;
public:
    MatrixBoardcastView(MatrixBase<ElementTy> matrix, int newRow, int newCol);

    ElementTy get(int row, int col) override;
    int getColumnCount() override { return newCol; }
    int getRowCount() override { return newRow; }
};

template <class ElementTy=float>
class MatrixTransposeView : public MatrixBase<ElementTy> {
public:
    MatrixTransposeView(MatrixBase<ElementTy> matrix);

    ElementTy get(int row, int col) override;
    int getColumnCount() override;
    int getRowCount() override;
};

template <class ElementTy=float>
class MatrixReshapeView : public MatrixBase<ElementTy> {
public:
    MatrixReshapeView(MatrixBase<ElementTy> matrix, int new_row, int new_column);
};

template <class ElementTy=float>
class RowView : public VectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(col); }

public:
    int row;
    RowView(MatrixBase<ElementTy> matrix, int row): VectorView<ElementTy>(matrix), row(row) { }

    ElementTy get(int col) override { return MatrixBase<ElementTy>::get(row, col); }
    ElementTy operator[] (int col) { return get(col); }
    int getColumnCount() override { return MatrixBase<ElementTy>::getColumnCount(); }
    int getRowCount() override { return 1; }
};

template <class ElementTy=float>
class MutableRowView : public MutableVectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(col); }

public:
    int row;
    MutableRowView(MatrixBase<ElementTy> matrix, int row): MutableVectorView<ElementTy>(matrix), row(row) { }

    ElementTy get(int col) override { return MatrixBase<ElementTy>::get(row, col); }
    ElementTy& operator[] (int col) { return *(MatrixBase<ElementTy>::data.get() + (row * MatrixBase<ElementTy>::numCol) + col); };

    void set(int index, ElementTy value) override { MutableVectorView<ElementTy>::set_elem(row, index, value); }
    int getColumnCount() override { return MatrixBase<ElementTy>::getColumnCount(); }
    int getRowCount() override { return 1; }

    MutableRowView<ElementTy>& operator=(MutableRowView<ElementTy>& another);
};

template <class ElementTy=float>
class ColumnView : public VectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(row); }

public:
    int col;
    ColumnView(MatrixBase<ElementTy> matrix, int col): VectorView<ElementTy>(matrix), col(col) { }
    ElementTy get(int row) { return MatrixBase<ElementTy>::get(row, col); }
    int getColumnCount() override { return 1; }
    int getRowCount() override { return MatrixBase<ElementTy>::getRowCount(); }
};

template <class ElementTy=float>
class MutableColumnView : public MutableVectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(row); }

public:
    int col;
    MutableColumnView(MatrixBase<ElementTy> matrix, int col): MutableVectorView<ElementTy>(matrix), col(col) { }
    void set(int index, ElementTy value) override { MutableVectorView<ElementTy>::set_elem(index, col, value); }
    ElementTy get(int row) { return MatrixBase<ElementTy>::get(row, col); }
    int getColumnCount() override { return 1; }
    int getRowCount() override { return MatrixBase<ElementTy>::getRowCount(); }
};


class LogicalException : public std::exception {
public:
    const char* msg;
    LogicalException(const char* msg): msg(msg) { }
    const char * what() const noexcept override { return msg; }
};

typedef ColumnView<> column_view_t;
typedef MutableColumnView<> mutable_column_view_t;
typedef RowView<> row_view_t;
typedef MutableRowView<> mutable_row_view_t;

///////////////////////////////////////////////////////////////////////////

template<class ElementTy>
MutableRowView<ElementTy> &MutableRowView<ElementTy>::operator=(MutableRowView<ElementTy> &another) {
    if (this->getColumnCount() != another.getColumnCount())
        throw LogicalException("Column count is not identical");
    int ncol = getColumnCount();
    for (int i=0; i<ncol; i++) {
        this->set(i, another.get(i));
    }
    return *this;
}


template<class ElementTy>
MatrixReshapeView<ElementTy>::MatrixReshapeView(MatrixBase<ElementTy> matrix, int new_row, int new_column) : MatrixBase<ElementTy>(matrix) {
    int elem_count = matrix.getElemCount();
    if (new_row < 1 && new_column < 1)
        throw LogicalException("at least one dimension should given");
    if (new_row < 1) new_row = elem_count / new_column;
    else if (new_column < 1) new_column = elem_count / new_row;
    if (new_row * new_column != elem_count)
        throw LogicalException("cannot reshape to given shape");
    this->numCol = new_column;
    this->numRow = new_row;
}

using matrix_t = Matrix<>;

template<class ElementTy>
MatrixBase<ElementTy>::MatrixBase(const MatrixBase &another): numCol(another.numCol), numRow(another.numRow), data(another.data) { }

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator+(Matrix<ElementTy> &another) {
    return boardcastBinaryOperator(another, [] (int a, int b) { return a + b; });
}

template<class ElementTy>
void MatrixBase<ElementTy>::print(const char *dstr) {
    int r = getRowCount(), c = getColumnCount();
    for (int i=0; i<r; i++) {
        for (int j = 0; j < c; j++)
            std::cout << get(i, j) << (dstr ? dstr : ", ");
        std::cout << "\n";
    }
    std::cout << "\n";
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::clone() {
    int r = getRowCount(), c = getColumnCount();
    Matrix<ElementTy> mat(r, c);
    for (int i=0; i<r; i++)
        for (int j = 0; j < c; j++)
            mat.set(i, j, this->get(i, j));
    return mat;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::transpose() {
    return MatrixTransposeView<ElementTy>(*this).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::resize(int r, int c) {
    return MatrixBoardcastView<ElementTy>(*this, r, c).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::reshape(int r, int c) {
    return MatrixReshapeView<ElementTy>(*this, r, c).clone();
}

template<class ElementTy>
MatrixBase<ElementTy> MatrixBase<ElementTy>::boardcastTo(int r, int c) {
    return MatrixReshapeView<ElementTy>(*this, r, c);
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::boardcastBinaryOperator(Matrix<ElementTy> &another,
                                                                 const std::function<ElementTy(ElementTy, ElementTy)> &op_handler) {
    int maxCol = max(this->getColumnCount(), another.getColumnCount());
    int maxRow = max(this->getRowCount(), another.getRowCount());
    MatrixBoardcastView<ElementTy>* mv1 = new MatrixBoardcastView<ElementTy>(*this, maxRow, maxCol);
    MatrixBoardcastView<ElementTy>* mv2 = new MatrixBoardcastView<ElementTy>(another, maxRow, maxCol);
    Matrix m(maxRow, maxCol);
    for (int i=0; i<maxRow; i++)
        for (int j=0; j<maxCol; j++) {
            ElementTy result = op_handler(mv1->get(i, j), mv2->get(i, j));
            //cout << (mv1->get(i, j) + mv2->get(i, j)) << "\n";
            m.set(i, j, result);
        }
    delete mv1, mv2;
    return m;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator-(Matrix<ElementTy> &another) {
    return boardcastBinaryOperator(another, [](int a, int b) { return a - b; });
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::dot(Matrix<ElementTy> &another) {
    return boardcastBinaryOperator(another, [](int a, int b) { return a * b; });
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator*(Matrix<ElementTy> &another) {
    int maxMid = max(this->getColumnCount(), another.getRowCount());

    MatrixBoardcastView<ElementTy>* mv1 = new MatrixBoardcastView<ElementTy>(*this, this->getRowCount(), maxMid);
    MatrixBoardcastView<ElementTy>* mv2 = new MatrixBoardcastView<ElementTy>(another, maxMid, another.getColumnCount());

    size_t new_row = this->getRowCount();
    size_t new_col = another.getColumnCount();
    Matrix m(new_row, new_col);
    //printf("max_mid=%d, new_row=%d, new_col=%d\n", maxMid, new_row, new_col);

    for (int i=0; i<new_row; i++) {
        for (int j=0; j<new_col; j++) {
            int result = 0;
            for (int k=0; k<maxMid; k++) {
                //printf("%f * %f = %f\n", get(i, k), another.get(k, j), get(i, k) * another.get(k, j));
                result += this->get(i, k) * another.get(k, j);
            }
            m.set(i, j, result);
        }
    }
    return m;
}

template<class ElementTy>
RowView<ElementTy> MatrixBase<ElementTy>::getRow(int row) {
    return RowView<ElementTy>(*this, row);
}

template<class ElementTy>
ColumnView<ElementTy> MatrixBase<ElementTy>::getColumn(int col) {
    return ColumnView<ElementTy>(*this, col);
}

template<class ElementTy>
RowView<ElementTy> MatrixBase<ElementTy>::operator[](int row) {
    return std::move(getRow(row));
}

template<class ElementTy>
Matrix<ElementTy>::Matrix(ElementTy number) {
    ElementTy* _data = new ElementTy[1];
    MatrixBase<ElementTy>::data = _data;
    *_data = number;
    MatrixBase<ElementTy>::numCol = 1;
    MatrixBase<ElementTy>::numRow = 1;
}

template<class ElementTy>
Matrix<ElementTy>::Matrix(int nr, int nc) {
    ElementTy* _data = new ElementTy[nc * nr];
    MatrixBase<ElementTy>::data.reset(_data);
    MatrixBase<ElementTy>::numCol = nc;
    MatrixBase<ElementTy>::numRow = nr;
    if constexpr(std::is_convertible_v<ElementTy, int>) {
        memset(_data, 0, sizeof(ElementTy) * nc * nr);
    }
}

template<class ElementTy>
void Matrix<ElementTy>::set(int row, int col, ElementTy value) {
    *(MatrixBase<ElementTy>::data.get() + (row * MatrixBase<ElementTy>::numCol) + col) = value;
}

template<class ElementTy>
Matrix<ElementTy>::~Matrix() { }

template<class ElementTy>
Matrix<ElementTy>::Matrix(const std::initializer_list<ElementTy> &data)
    : Matrix({ data }) { }

template<class ElementTy>
Matrix<ElementTy>::Matrix(const std::initializer_list<std::initializer_list<ElementTy>> &data) {
    int nc = data.begin()->size();
    int nr = data.size();
    ElementTy* _data = new ElementTy[nc * nr];
    MatrixBase<ElementTy>::data.reset(_data);
    MatrixBase<ElementTy>::numCol = nc;
    MatrixBase<ElementTy>::numRow = nr;
    int idx = 0;
    for (auto it1=data.begin(); it1 != data.end(); it1++)
        for (auto it2=it1->begin(); it2 != it1->end(); it2++)
            _data[idx++] = *it2;
}

template<class ElementTy>
void Matrix<ElementTy>::swapRow(int row1, int row2) {
    int col_count = this->numCol;
    for (int i=0; i<col_count; i++)
        swapElem(row1, i, row2, i);
}

template<class ElementTy>
void Matrix<ElementTy>::swapCol(int col1, int col2) {
    int row_count = this->numRow;
    for (int i=0; i<row_count; i++)
        swapElem(i, col1, i, col2);
}

template<class ElementTy>
void Matrix<ElementTy>::swapElem(int col1, int row1, int col2, int row2) {
    ElementTy el1 = this->get(row1, col1);
    ElementTy el2 = this->get(row2, col2);
    this->set(row1, col1, el2);
    this->set(col2, row2, el1);
}

template<class ElementTy>
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
Matrix<ElementTy>::zeros(int r, int c) {
    return std::move(Matrix<ElementTy>());
}

template<class ElementTy>
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
Matrix<ElementTy>::ones(int r, int c) {
    Matrix<ElementTy> mat(r, c);
    for (int i=0; i<r*c; i++)
        mat.data[i] = 1;
    return mat;
}

template<class ElementTy>
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>
Matrix<ElementTy>::identity(int size, ElementTy value) {
    Matrix<ElementTy> mat(size, size);
    for (int i=0; i<size; i++)
        mat.set(i, i, value);
    return mat;
}

template<class ElementTy>
MutableRowView<ElementTy> Matrix<ElementTy>::getMutableRow(int row) {
    return MutableRowView<ElementTy>(*this, row);
}

template<class ElementTy>
MutableColumnView<ElementTy> Matrix<ElementTy>::getMutableColumn(int col) {
    return MutableColumnView<ElementTy>(*this, col);
}

template<class ElementTy>
MatrixBoardcastView<ElementTy>::MatrixBoardcastView(MatrixBase<ElementTy> matrix, int newRow, int newCol)
        : MatrixBase<ElementTy>(matrix), scaleRow(newRow * 1.0 / matrix.getRowCount()), scaleCol(newCol * 1.0 / matrix.getColumnCount()),
          newCol(newCol), newRow(newRow) { }

template<class ElementTy>
ElementTy MatrixBoardcastView<ElementTy>::get(int row, int col) {
    //printf("Mapping (%d, %d) to (%d, %d)\n", row, col, (int)(1.0 * row / scaleRow), (int)(1.0 * col / scaleCol));
    return MatrixBase<ElementTy>::get(1.0 * row / scaleRow, 1.0 * col / scaleCol);
}

template<class ElementTy>
MatrixTransposeView<ElementTy>::MatrixTransposeView(MatrixBase<ElementTy> matrix): MatrixBase<ElementTy>(matrix) { }

template<class ElementTy>
ElementTy MatrixTransposeView<ElementTy>::get(int row, int col) {
    return MatrixBase<ElementTy>::get(col, row);
}

template<class ElementTy>
int MatrixTransposeView<ElementTy>::getColumnCount() {
    return this->numRow; // return thr real row number since transposed
}

template<class ElementTy>
int MatrixTransposeView<ElementTy>::getRowCount() {
    return this->numCol; // return thr real column number since transposed
}

#endif //LIBNUMANALYSIS_MATRIX_H
