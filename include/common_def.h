#ifndef LIBNUMANALYSIS_COMMONDEF_H
#define LIBNUMANALYSIS_COMMONDEF_H

#define MAX_ITERATION_COUNT 300

#define MATRIX_API       template <class ElementTy>
#define MATRIX_T         Matrix<ElementTy>
#define MATRIX_FROM_VIEW(view_name, mat) view_name<ElementTy>(mat).clone()
#define ELEMENT_T        ElementTy

#define EQU_API          template <class ValTy>
#define VALUE_T          ValTy

EQU_API
using SingleMetaFunction = ValTy(*)(ValTy v);
using single_meta_function_t = SingleMetaFunction<float>;
#define SINGLE_META_FUNCTION_T SingleMetaFunction<ValTy>

EQU_API
using DualMetaFunction = ValTy(*)(ValTy x, ValTy y);
using dual_meta_function_t = DualMetaFunction<float>;
#define DUAL_META_FUNCTION_T DualMetaFunction<ValTy>

enum iter_result_t {
    ITER_OK = 0,                      // iteration success
    ITER_UNCOVERAGED = 1,             // equation is uncoveraged
    ITER_REQUIRE_SQUARE_MATRIX = 2,   // require a square matrix for input
    ITER_INVADE_ARG = 3,              // invalid argument
};


#endif //LIBNUMANALYSIS_COMMONDEF_H
