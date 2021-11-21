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
#include <random>
#include <iomanip>

#define PERIOR_DEFINE_CLASS(class_name) template <class T> class class_name

PERIOR_DEFINE_CLASS(Matrix);
PERIOR_DEFINE_CLASS(ColumnView);
PERIOR_DEFINE_CLASS(RowView);
PERIOR_DEFINE_CLASS(MutableRowView);
PERIOR_DEFINE_CLASS(MutableColumnView);

template <class ElementTy=float>
class MatrixBase {
public:
    class Formatter {
    public:
        virtual int printElement(std::ostream& os, ElementTy& el, int row, int col, int rowNum, int colNum) = 0;
        virtual int printLineStart(std::ostream& os, int row, int rowNum, int colNum) = 0;
        virtual int printLineEnd(std::ostream& os, int row, int rowNum, int colNum) = 0;
        virtual int printMatrixStart(std::ostream& os) = 0;
        virtual int printEnd(std::ostream& os) = 0;
    };
protected:
    class DefaultFormatter : public Formatter {
        int printElement(std::ostream &os, ElementTy &el, int row, int col, int rowNum, int colNum) override;
        int printLineStart(std::ostream &os, int row, int rowNum, int colNum) override;
        int printLineEnd(std::ostream &os, int row, int rowNum, int colNum) override;
        int printMatrixStart(std::ostream &os) override;
        int printEnd(std::ostream &os) override;
    };
    DefaultFormatter defaultFormatter;
    Formatter* _commonFormatter = nullptr;
public:
    void setFormatter(Formatter* fmt) { _commonFormatter = fmt; }
    Formatter* getFormatter() { return _commonFormatter; }
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
    // set value of an element
    void _set_elem(int row, int col, ElementTy value) { *(data.get() + (row * numCol) + col) = value; }
    ElementTy _get_elem(int row, int col) { return *(data.get() + (row * numCol) + col); };
    ElementTy& _get_elem_ref(int row, int col) { return *(data.get() + (row * numCol) + col); };

public:
    // reference constructor
    MatrixBase(const MatrixBase& another);
    // Get element at given row and column
    virtual ElementTy get(int row, int col) { return *(data.get() + (row * numCol) + col); }
    // Print the matrix
    void print(std::ostream &os = std::cout, MatrixBase::Formatter *formatter = nullptr);
    // Print the matrix
    friend std::ostream& operator<< (std::ostream& os, const MatrixBase<ElementTy>& mat) { const_cast<MatrixBase<ElementTy>*>(&mat)->print(os); return os; }
    // Perform broadcast add
    Matrix<ElementTy> operator+ (const MatrixBase<ElementTy>& another) ;
    // Perform broadcast substract
    Matrix<ElementTy> operator- (const MatrixBase<ElementTy>& another) ;
    // Perform broadcast multiply
    Matrix<ElementTy> operator* (const MatrixBase<ElementTy>& another) ;
    // Perform broadcast dot multiply
    Matrix<ElementTy> dot(const MatrixBase<ElementTy>& another) ;
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
    Matrix<ElementTy> boardcastBinaryOperator(const MatrixBase<ElementTy>& another, const std::function<ElementTy (ElementTy, ElementTy)>& op_handler);
    // get a row
    RowView<ElementTy> getRow(int row);
    // get a row
    RowView<ElementTy> operator[] (int row);
    // get a column
    ColumnView<ElementTy> getColumn(int col);
    // get raw data in memory
    virtual ElementTy* rawData() { return data.get(); }
    // get upper triangle matrix
    Matrix<ElementTy> upperTriangle();
    // get lower triangle matrix
    Matrix<ElementTy> lowerTriangle();
    // get diagonal matrix
    Matrix<ElementTy> diagonal();
    // create a new matrix woth same size to this matrix, with all elements are k
    Matrix<ElementTy> shapeLike(ElementTy k);
    // create a new matrix woth same size to this matrix, with all elements are 1
    Matrix<ElementTy> onesLike() { return shapeLike(1); }
    // create a new matrix woth same size to this matrix, with all elements are 0
    Matrix<ElementTy> zerosLike() { return shapeLike(0); }
    // get negative value
    Matrix<ElementTy> operator- ();
    // get positive value
    Matrix<ElementTy> operator+ () { return clone(); }
    // get absoulte value of each element
    Matrix<ElementTy> abs();
    // get L1 norm of matrix
    ElementTy l1Norm();
    // get L2 norm of matrix
    ElementTy l2Norm();
    // get infinte norm of matrix
    ElementTy infNorm();
    // get sum of matrix
    ElementTy sum();
    // get max element
    ElementTy max();
    // get argmax element
    ElementTy argmax();
    // get max of given dimension
    Matrix<ElementTy> max(int dim);
    // get argmax of given dimension
    Matrix<ElementTy> argmax(int dim);
    // get inverse matrix
    Matrix<ElementTy> inverse() { return std::move(inverseGaussJordan()); }
    // get inverse matrix via Gauss-Jordan method
    Matrix<ElementTy> inverseGaussJordan();
    // divide a number
    Matrix<ElementTy> operator / (ElementTy n);
    // times a number
    Matrix<ElementTy> operator * (ElementTy n);
    // get shape of the matrix
    Matrix<ElementTy> shape() { return Matrix<ElementTy>({ (ElementTy)this->getRowCount(), (ElementTy)this->getColumnCount() }); }

    virtual ~MatrixBase() = default;

protected:
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
    // intiialize with an empty matrix, means a placeholder for matrix, no operation can be performed
    Matrix() = default;
    // set value of an element
    void set(int row, int col, ElementTy value) override;
    // swap two rows
    void swapRow(int row1, int row2);
    // swap two columns
    void swapCol(int col1, int col2);
    // swap two elements
    void swapElem(int col1, int row1, int col2, int row2);
    // get all-zero matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
    zeros(int r, int c);
    // get all-one matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
    ones(int r, int c);
    // get identity matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
    identity(int size, ElementTy value=1);
    // get random matrix
    static typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
    randn(int r, int c);
    // get a row
    MutableRowView<ElementTy> getMutableRow(int row);
    // get a row
    MutableRowView<ElementTy> operator[] (int row) { return std::move(this->getMutableRow(row)); }
    // get a column
    MutableColumnView<ElementTy> getMutableColumn(int col);

    ~Matrix();
};

class LogicalException : public std::exception {
public:
    const char* msg;
    LogicalException(const char* msg): msg(msg) { }
    const char * what() const noexcept override { return msg; }
};

#define REQUIRE_SQUARE_MATRIX() REQUIRE_SQUARE_MATRIX_(matrix)

#define REQUIRE_SQUARE_MATRIX_(mat) \
        do {\
            if (const_cast<MatrixBase<ElementTy>*>(&(mat))->getRowCount() != const_cast<MatrixBase<ElementTy>*>(&(mat))->getColumnCount())\
                throw LogicalException("require a square matrix");\
        } while (0)

template <class ElementTy=float>
class VectorView : public MatrixBase<ElementTy> {
public:
    VectorView(const MatrixBase<ElementTy>& matrix) : MatrixBase<ElementTy>(matrix) { }
    virtual ElementTy get(int index) = 0;
};

template <class ElementTy=float>
class MutableVectorView : public MatrixBase<ElementTy> {
protected:
    void set_elem(int row, int col, ElementTy value) {
        *(MatrixBase<ElementTy>::data.get() + (row * MatrixBase<ElementTy>::numCol) + col) = value;
    }
public:
    MutableVectorView(const MatrixBase<ElementTy>& matrix) : MatrixBase<ElementTy>(matrix) { }
    virtual ElementTy get(int index) = 0;
    virtual void set(int index, ElementTy value) = 0;
};

template <class ElementTy=float>
class MatrixBoardcastView : public MatrixBase<ElementTy> {
    float scaleRow, scaleCol;
    int newCol, newRow;
public:
    MatrixBoardcastView(const MatrixBase<ElementTy>& matrix, int newRow, int newCol);

    ElementTy get(int row, int col) override;
    int getColumnCount() override { return newCol; }
    int getRowCount() override { return newRow; }
};

template <class ElementTy=float>
class MatrixTransposeView : public MatrixBase<ElementTy> {
public:
    MatrixTransposeView(const MatrixBase<ElementTy>& matrix);

    ElementTy get(int row, int col) override;
    int getColumnCount() override;
    int getRowCount() override;
};

template <class ElementTy=float>
class MatrixReshapeView : public MatrixBase<ElementTy> {
public:
    MatrixReshapeView(const MatrixBase<ElementTy>& matrix, int new_row, int new_column);
};

template <class ElementTy=float>
class UpperTriangularMatrixView : public MatrixBase<ElementTy> {
public:
    UpperTriangularMatrixView(const MatrixBase<ElementTy>& matrix): MatrixBase<ElementTy>(matrix) {
        REQUIRE_SQUARE_MATRIX();
    }

    ElementTy get(int row, int col) override { return col <= row ? 0 : MatrixBase<ElementTy>::_get_elem(row, col); }
};

template <class ElementTy=float>
class LowerTriangularMatrixView : public MatrixBase<ElementTy> {
public:
    LowerTriangularMatrixView(const MatrixBase<ElementTy>& matrix): MatrixBase<ElementTy>(matrix) {
        REQUIRE_SQUARE_MATRIX();
    }

    ElementTy get(int row, int col) override { return col >= row ? 0 : MatrixBase<ElementTy>::_get_elem(row, col); }
};

template <class ElementTy=float>
class DiagonalMatrixView : public MatrixBase<ElementTy> {
public:
    DiagonalMatrixView(const MatrixBase<ElementTy>& matrix): MatrixBase<ElementTy>(matrix) {
        REQUIRE_SQUARE_MATRIX();
    }

    ElementTy get(int row, int col) override { return col != row ? 0 : MatrixBase<ElementTy>::_get_elem(row, col); }
};

template <class ElementTy=float>
class KsLikeMatrixView : public MatrixBase<ElementTy> {
public:
    ElementTy k;
    KsLikeMatrixView(const MatrixBase<ElementTy>& matrix, ElementTy k): MatrixBase<ElementTy>(matrix), k(k) { }

    ElementTy get(int row, int col) override { return k; }
};

template <class ElementTy=float, ElementTy (*FuncPtr)(ElementTy, int, int)=nullptr>
class ElementWiseMatrixView : public MatrixBase<ElementTy> {
public:
    ElementWiseMatrixView(const MatrixBase<ElementTy>& matrix): MatrixBase<ElementTy>(matrix) { }

    ElementTy get(int row, int col) override { return FuncPtr(this->_get_elem(row, col), row, col); }
};

#define DEFINE_ELEMENTWISE_OPERATOR_VIEW(class_name, operation) \
template <class ElementTy> \
ElementTy _CallFunc_##class_name(ElementTy x, int row, int col) { return operation; }\
\
template <class ElementTy = float>\
using class_name = ElementWiseMatrixView<ElementTy, _CallFunc_##class_name>;

DEFINE_ELEMENTWISE_OPERATOR_VIEW(NegativeMatrixView, -x)
DEFINE_ELEMENTWISE_OPERATOR_VIEW(ElementWiseReciprocalView, row == col ? 1/x : 0)
DEFINE_ELEMENTWISE_OPERATOR_VIEW(PositiveMatrixView, x)
DEFINE_ELEMENTWISE_OPERATOR_VIEW(AbsoluteMatrixView, x >= 0 ? x : -x)

template <class ElementTy=float, ElementTy (*FuncPtr)(ElementTy, ElementTy, int, int)=nullptr>
class ElementWiseNumbericOperateView : public MatrixBase<ElementTy> {
public:
    ElementTy k;
    ElementWiseNumbericOperateView(const MatrixBase<ElementTy>& matrix, ElementTy k): MatrixBase<ElementTy>(matrix), k(k) { }

    ElementTy get(int row, int col) override { return FuncPtr(this->_get_elem(row, col), k, row, col); }
};

#define DEFINE_ELEMENTWISE_NUMBERIC_OPERATOR_VIEW(class_name, operation) \
template <class ElementTy> \
ElementTy _CallFunc_##class_name(ElementTy x, ElementTy n, int row, int col) { return operation; }\
\
template <class ElementTy = float>\
using class_name = ElementWiseNumbericOperateView<ElementTy, _CallFunc_##class_name>;

DEFINE_ELEMENTWISE_NUMBERIC_OPERATOR_VIEW(MatrixNumbericDivideView, x / n)
DEFINE_ELEMENTWISE_NUMBERIC_OPERATOR_VIEW(MatrixNumbericTimesView, x * n)

template <class ElementTy=float>
class RowView : public VectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(col); }

public:
    int row;
    RowView(const MatrixBase<ElementTy>& matrix, int row): VectorView<ElementTy>(matrix), row(row) { }

    ElementTy get(int col) override { return MatrixBase<ElementTy>::get(row, col); }
    ElementTy operator[] (int col) { return get(col); }
    int getColumnCount() override { return MatrixBase<ElementTy>::getColumnCount(); }
    int getRowCount() override { return 1; }
    //ElementTy* rawData() override { return this->getRawData(row, 0); }
};

template <class ElementTy=float>
class MutableRowView : public MutableVectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(col); }

public:
    int row;
    MutableRowView(const MatrixBase<ElementTy>& matrix, int row): MutableVectorView<ElementTy>(matrix), row(row) { }

    ElementTy get(int col) override { return MatrixBase<ElementTy>::get(row, col); }
    ElementTy& operator[] (int col) { return *(MatrixBase<ElementTy>::data.get() + (row * MatrixBase<ElementTy>::numCol) + col); };

    void set(int index, ElementTy value) override { MutableVectorView<ElementTy>::set_elem(row, index, value); }
    void set(const MatrixBase<ElementTy>& another);
    MutableRowView<ElementTy>& operator=(const MatrixBase<ElementTy>& another) { set(another); return *this; }
    int getColumnCount() override { return MatrixBase<ElementTy>::getColumnCount(); }
    int getRowCount() override { return 1; }
    MutableRowView<ElementTy>& operator=(MutableRowView<ElementTy>& another);
    //ElementTy* rawData() override { return this->getRawData(row, 0); }
};

template <class ElementTy=float>
class ColumnView : public VectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(row); }
    ElementTy* rawData() override { return nullptr; }

public:
    int col;
    ColumnView(const MatrixBase<ElementTy>& matrix, int col): VectorView<ElementTy>(matrix), col(col) { }
    ElementTy get(int row) { return MatrixBase<ElementTy>::get(row, col); }
    int getColumnCount() override { return 1; }
    int getRowCount() override { return MatrixBase<ElementTy>::getRowCount(); }
};

template <class ElementTy=float>
class MutableColumnView : public MutableVectorView<ElementTy> {
private:
    ElementTy get(int row, int col) override { return get(row); }
    ElementTy* rawData() override { return nullptr; }

public:
    int col;
    MutableColumnView(const MatrixBase<ElementTy>& matrix, int col): MutableVectorView<ElementTy>(matrix), col(col) { }
    void set(int index, ElementTy value) override { MutableVectorView<ElementTy>::set_elem(index, col, value); }
    ElementTy get(int row) { return MatrixBase<ElementTy>::get(row, col); }
    int getColumnCount() override { return 1; }
    int getRowCount() override { return MatrixBase<ElementTy>::getRowCount(); }
};


typedef ColumnView<> column_view_t;
typedef MutableColumnView<> mutable_column_view_t;
typedef RowView<> row_view_t;
typedef MutableRowView<> mutable_row_view_t;

#define PRINT_MAT(M) std::cout << #M << " = \n" << (M)

///////////////////////////////////////////////////////////////////////////


template<class ElementTy>
int MatrixBase<ElementTy>::DefaultFormatter::printElement(std::ostream &os, ElementTy &el, int row, int col, int rowNum,
                                                          int colNum) {
    os << std::left << std::setprecision(5) << std::setw(10) << el;
    return 1;
}

template<class ElementTy>
int MatrixBase<ElementTy>::DefaultFormatter::printLineStart(std::ostream &os, int row, int rowNum, int colNum) {
    if (row > 0) os << " ";
    os << "[";
    return 1;
}

template<class ElementTy>
int MatrixBase<ElementTy>::DefaultFormatter::printLineEnd(std::ostream &os, int row, int rowNum, int colNum) {
    os << "]";
    if (row != rowNum - 1) os << "\n";
    return 1;
}

template<class ElementTy>
int MatrixBase<ElementTy>::DefaultFormatter::printMatrixStart(std::ostream &os) {
    os << "[";
    return 1;
}

template<class ElementTy>
int MatrixBase<ElementTy>::DefaultFormatter::printEnd(std::ostream &os) {
    os << "]\n";
    return 0;
}

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
void MutableRowView<ElementTy>::set(const MatrixBase<ElementTy> &another) {
    MatrixBase<ElementTy>* ano = const_cast<MatrixBase<ElementTy>*>(&another);
    if (ano->getColumnCount() != this->getColumnCount())
        throw LogicalException("row count are not the same");
    if (ano->getRowCount() != 1)
        throw LogicalException("aother row count is not 1");
    int n = this->getColumnCount();
    for (int i=0; i<n; i++)
        this->set(i, ano->get(0, i));
}


template<class ElementTy>
MatrixReshapeView<ElementTy>::MatrixReshapeView(const MatrixBase<ElementTy>& matrix, int new_row, int new_column) : MatrixBase<ElementTy>(matrix) {
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
MatrixBase<ElementTy>::MatrixBase(const MatrixBase &another):
    numCol(const_cast<MatrixBase<ElementTy>*>(&another)->getColumnCount()),
    numRow(const_cast<MatrixBase<ElementTy>*>(&another)->getRowCount()) {
        data = another.data;
    }

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator+(const MatrixBase<ElementTy>& another) {
    return boardcastBinaryOperator(another, [] (ElementTy a, ElementTy b) { return a + b; });
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::clone() {
    int r = this->getRowCount(), c = this->getColumnCount();
    Matrix<ElementTy> mat(r, c);
    for (int i=0; i<r; i++)
        for (int j = 0; j < c; j++)
            mat.set(i, j, this->get(i, j));
    return mat;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::transpose() {
    return MatrixTransposeView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::resize(int r, int c) {
    return MatrixBoardcastView<ElementTy>(this->clone(), r, c).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::reshape(int r, int c) {
    return MatrixReshapeView<ElementTy>(this->clone(), r, c).clone();
}

template<class ElementTy>
MatrixBase<ElementTy> MatrixBase<ElementTy>::boardcastTo(int r, int c) {
    return MatrixReshapeView<ElementTy>(this->clone(), r, c);
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::boardcastBinaryOperator(const MatrixBase<ElementTy>& another,
                                                                 const std::function<ElementTy(ElementTy, ElementTy)> &op_handler) {
    int maxCol = max(this->getColumnCount(), ((MatrixBase<ElementTy>*)&another)->getColumnCount());
    int maxRow = max(this->getRowCount(), ((MatrixBase<ElementTy>*)&another)->getRowCount());
    MatrixBoardcastView<ElementTy>* mv1 = new MatrixBoardcastView<ElementTy>(*this, maxRow, maxCol);
    MatrixBoardcastView<ElementTy>* mv2 = new MatrixBoardcastView<ElementTy>(another, maxRow, maxCol);
    Matrix m(maxRow, maxCol);
    for (int i=0; i<maxRow; i++)
        for (int j=0; j<maxCol; j++) {
            ElementTy result = op_handler(mv1->get(i, j), mv2->get(i, j));
            //std::cout << (mv1->get(i, j) + mv2->get(i, j)) << "\n";
            m.set(i, j, result);
        }
    delete mv1, mv2;
    return m;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator-(const MatrixBase<ElementTy>& another) {
    return boardcastBinaryOperator(another, [](ElementTy a, ElementTy b) { return a - b; });
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::dot(const MatrixBase<ElementTy>& another) {
    return boardcastBinaryOperator(another, [](ElementTy a, ElementTy b) { return a * b; });
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator*(const MatrixBase<ElementTy>& another) {
    int maxMid = max(this->getColumnCount(), const_cast<MatrixBase<ElementTy>*>(&another)->getRowCount());

    MatrixBoardcastView<ElementTy>* mv1 = new MatrixBoardcastView<ElementTy>(*this, this->getRowCount(), maxMid);
    MatrixBoardcastView<ElementTy>* mv2 = new MatrixBoardcastView<ElementTy>(*const_cast<MatrixBase<ElementTy>*>(&another), maxMid, const_cast<MatrixBase<ElementTy>*>(&another)->getColumnCount());

    size_t new_row = this->getRowCount();
    size_t new_col = const_cast<MatrixBase<ElementTy>*>(&another)->getColumnCount();
    Matrix m(new_row, new_col);
    //printf("max_mid=%d, new_row=%d, new_col=%d\n", maxMid, new_row, new_col);

    for (int i=0; i<new_row; i++) {
        for (int j=0; j<new_col; j++) {
            ElementTy result = 0;
            for (int k=0; k<maxMid; k++) {
                //printf("%f * %f = %f\n", get(i, k), another.get(k, j), get(i, k) * another.get(k, j));
                result += this->get(i, k) * const_cast<MatrixBase<ElementTy>*>(&another)->get(k, j);
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
Matrix<ElementTy> MatrixBase<ElementTy>::upperTriangle() {
    return UpperTriangularMatrixView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::lowerTriangle() {
    return LowerTriangularMatrixView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::diagonal() {
    return DiagonalMatrixView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator-() {
    return NegativeMatrixView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::shapeLike(ElementTy k) {
    return KsLikeMatrixView<ElementTy>(this->clone(), k).clone();
}

template<class ElementTy>
void MatrixBase<ElementTy>::print(std::ostream &os, MatrixBase::Formatter *formatter) {
    if (formatter == nullptr)
        formatter = _commonFormatter;
    if (formatter == nullptr)
        formatter = &defaultFormatter;
    int r = getRowCount(), c = getColumnCount();
    formatter->printMatrixStart(os);
    for (int i=0; i<r; i++) {
        formatter->printLineStart(os, i, r, c);
        for (int j = 0; j < c; j++) {
            ElementTy val = std::move(get(i, j));
            formatter->printElement(os, val, i, j, r, c);
        }
        formatter->printLineEnd(os, i, r, c);
    }
    formatter->printEnd(os);
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::abs() {
    return AbsoluteMatrixView<ElementTy>(this->clone()).clone();
}

template<class ElementTy>
ElementTy MatrixBase<ElementTy>::sum() {
    int r = this->getRowCount(), c = this->getColumnCount();
    ElementTy sum = 0;
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++)
            sum += this->get(i, j);
    }
    return sum;
}

template<class ElementTy>
ElementTy MatrixBase<ElementTy>::max() {
    int r = this->getRowCount(), c = this->getColumnCount();
    ElementTy _max = this->get(0, 0);
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++)
            _max = max(_max, this->get(i, j));
    }
    return _max;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::max(int dim) {
    int n;
    // calc max on rows, returs rows
    if (dim == 0) {
        n = this->getRowCount();
        Matrix<ElementTy> mat(1, n);
        //printf("n=%d\n", n);
        for (int i=0; i<n; i++) {
            this->getRow(i).print();
            ElementTy max_val = this->getRow(i).max();
            mat[0][i] = max_val;
        }
        return mat;
    }
    // calc max on cols, returns cols
    else if (dim == 1) {
        n = this->getColumnCount();
        Matrix<ElementTy> mat(1, n);
        for (int i=0; i<n; i++) {
            ElementTy max_val = this->getColumn(i).max();
            mat[0][i] = max_val;
        }
        return mat;
    }
    throw LogicalException("no such dimension");
}

template<class ElementTy>
ElementTy MatrixBase<ElementTy>::l1Norm() {
    int n = this->getColumnCount();
    ElementTy _max = this->getColumn(0).sum();
    for (int i=1; i<n; i++) {
        _max = max(_max, this->getColumn(i).sum());
    }
    return _max;
}

template<class ElementTy>
ElementTy MatrixBase<ElementTy>::infNorm() {
    int n = this->getRowCount();
    ElementTy _max = this->getRow(0).sum();
    for (int i=1; i<n; i++) {
        _max = max(_max, this->getRow(i).sum());
    }
    return _max;
}

template<class ElementTy>
ElementTy MatrixBase<ElementTy>::l2Norm() {
    // if is vector, calculate vector
    int c = this->getColumnCount(), r = this->getRowCount();
    MatrixBase<ElementTy>* mat = this;
    if (c == 1 || r == 1) {
        // if is vertical vector, transpose it
        auto transposedMat = MatrixTransposeView<ElementTy>(*this);
        if (c == 1)
            mat = &transposedMat;
        ElementTy sum = 0;
        for (int i=0; i<c; i++) {
            ElementTy val = mat->get(0, i);
            sum += val * val;
        }
        return sqrt(sum);
    }
    // calc L2 norm of matrix using \lambdaE - A = 0
    throw "unimplemented";
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::inverseGaussJordan() {
    REQUIRE_SQUARE_MATRIX_(*this);
    int n = this->getColumnCount();
    Matrix<ElementTy> A = this->clone();
    Matrix<ElementTy> E = Matrix<ElementTy>::identity(n);
    //PRINT_MAT(A);
    //PRINT_MAT(E);
    // (A|E) = (E|A^-1)
    for (int i=0; i<n; i++) {
        ElementTy A_ii = A[i][i];
        A[i] = A[i] / A_ii; E[i] = E[i] / A_ii;
        for (int j=i+1; j<n; j++) {
            ElementTy A_ji = A[j][i];
            A[j] = A[j].clone() - A[i] * A_ji; E[j] = E[j].clone() - E[i] * A_ji;
        }
    }
    for (int i=n-1; i>=0; i--) {
        for (int j=i-1; j>=0; j--) {
            ElementTy A_ji = A[j][i];
            A[j] = A[j].clone() - A[i] * A_ji; E[j] = E[j].clone() - E[i] * A_ji;
        }
    }
    //PRINT_MAT(A);
    //PRINT_MAT(E);
    return E;
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator/(ElementTy n) {
    return MatrixNumbericDivideView<ElementTy>(this->clone(), n).clone();
}

template<class ElementTy>
Matrix<ElementTy> MatrixBase<ElementTy>::operator*(ElementTy n) {
    return MatrixNumbericTimesView<ElementTy>(this->clone(), n).clone();
}

template<class ElementTy>
Matrix<ElementTy>::Matrix(ElementTy number) {
    ElementTy* _data = new ElementTy[1];
    MatrixBase<ElementTy>::data.reset(_data);
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
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
Matrix<ElementTy>::zeros(int r, int c) {
    Matrix<ElementTy> mat(r, c);
    for (int i=0; i<r*c; i++)
        mat.data.get()[i] = 0;
    return mat;
}

template<class ElementTy>
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
Matrix<ElementTy>::ones(int r, int c) {
    Matrix<ElementTy> mat(r, c);
    for (int i=0; i<r*c; i++)
        mat.data.get()[i] = 1;
    return mat;
}

template<class ElementTy>
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
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
typename std::enable_if<std::is_convertible_v<ElementTy, int>, Matrix<ElementTy>>::type
Matrix<ElementTy>::randn(int r, int c) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist;

    Matrix<ElementTy> mat(r, c);
    ElementTy* data = mat.data.get();
    for (int i=0; i<r*c; i++)
        data[i] = dist(mt);
    return mat;
}

template<class ElementTy>
MatrixBoardcastView<ElementTy>::MatrixBoardcastView(const MatrixBase<ElementTy>& matrix, int newRow, int newCol)
        : MatrixBase<ElementTy>(matrix), scaleRow(newRow * 1.0 / ((MatrixBase<ElementTy>*)&matrix)->getRowCount()), scaleCol(newCol * 1.0 / ((MatrixBase<ElementTy>*)&matrix)->getColumnCount()),
          newCol(newCol), newRow(newRow) { }

template<class ElementTy>
ElementTy MatrixBoardcastView<ElementTy>::get(int row, int col) {
    //printf("Mapping (%d, %d) to (%d, %d)\n", row, col, (int)(1.0 * row / scaleRow), (int)(1.0 * col / scaleCol));
    return MatrixBase<ElementTy>::get(1.0 * row / scaleRow, 1.0 * col / scaleCol);
}

template<class ElementTy>
MatrixTransposeView<ElementTy>::MatrixTransposeView(const MatrixBase<ElementTy>& matrix): MatrixBase<ElementTy>(matrix) { }

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
