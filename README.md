# Numerical Analysis Library

**still working on**

This is a header-only library, just copy files in `include` directory to your project to use this libaray.

Sample usage (also see `example`):

```c++
#include <iostream>
#include "equation.h"

using namespace std;

int main() {
    try {
        matrix_t A = { {8,-3,2},{4,11,-1},{2,1,4} };
        matrix_t b = { 20,33,12 };
        matrix_t x0 = b.zerosLike();
        matrix_t x;
        jacobiInteration(A, b, 1e-6f, x);
        PRINT_MAT(x);
        gaussSeidelIteration(A, b, 1e-6f, x);
        PRINT_MAT(x);
        SORIteration(A, b, 1.0f, 1e-6f, x);
        PRINT_MAT(x);
        return 0;
    }
    catch (std::exception& err) {
        cout << err.what() << endl;
    }
}
```

The output would be:
```
x =
[[3         ]
 [2         ]
 [1         ]]
x =
[[3         ]
 [2         ]
 [1         ]]
x =
[[3         ]
 [2         ]
 [1         ]]
```

