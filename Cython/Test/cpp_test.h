#ifndef TEST_H
#define TEST_H

class Test {
public:
    int test1;
    Test();
    Test(int test1);
    ~Test();
    int returnFour();
    int returnFive();
    Test operator+(const Test& other);
    Test operator-(const Test& other);
};
#endif
