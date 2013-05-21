import numpy as np
import ctypes

square = ctypes.cdll.LoadLibrary("./square.so")

n = 5
a = np.arange(n, dtype=np.double)

aptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
square.square(aptr, n)

print a
