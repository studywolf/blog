cdef extern from "cpp_test.h":
    cdef cppclass Test:
        Test()
        Test(int test1)
        int test1
        int returnFive()
        Test add "operator+"(Test other)
        Test sub "operator-"(Test other)

cdef class pyTest:
    cdef Test* thisptr # hold a C++ instance
    def __cinit__(self, int test1):
        self.thisptr = new Test(test1)
    def __dealloc__(self):
        del self.thisptr

    def __add__(pyTest left, pyTest other):
        cdef Test t = left.thisptr.add(other.thisptr[0])
        cdef pyTest tt = pyTest(t.test1)
        return tt
    def __sub__(pyTest left, pyTest other):
        cdef Test t = left.thisptr.sub(other.thisptr[0])
        cdef pyTest tt = pyTest(t.test1)
        return tt

    def __repr__(self):
        return "pyTest[%s]" % (self.thisptr.test1)

    def returnFive(self):
        return self.thisptr.returnFive()

    def printMe(self):
        return "hello world"
