#include "cpp_test.h"

Test::Test() {
    test1 = 0;
}

Test::Test(int test1) {
    this->test1 = test1;
}

Test::~Test() { }

int Test::returnFour() { return 4; }

int Test::returnFive() { return returnFour() + 1; }

Test Test::operator+(const Test& other) {
    return Test(test1 + other.test1);
}

Test Test::operator-(const Test& other) {
    return Test(test1 - other.test1);
}
