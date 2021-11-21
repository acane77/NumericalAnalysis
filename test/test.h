#ifndef MIYUKI_TEST_H
#define MIYUKI_TEST_H

#include "gtest/gtest.h"

#define TEST_MAIN() \
int main(int argc, char** argv) {\
    testing::InitGoogleTest(&argc, argv);\
    return RUN_ALL_TESTS();\
}

#endif