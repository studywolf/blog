import matplotlib.pyplot as plt
import seaborn
import timeit


print('\nTest function 1: ')
time_sympy1 = timeit.timeit(
    stmt='f(np.random.random(), np.random.random())',
    setup='import numpy as np;\
           from sympy.utilities.autowrap import autowrap;\
           import sympy as sp;\
           q0 = sp.Symbol("q0");\
           l0 = sp.Symbol("l0");\
           a = sp.cos(q0) * l0;\
           f = autowrap(a, backend="cython", args=(q0, l0))')
print('Sympy autowrap function 1 time: ', time_sympy1)

time_hardcoded1 = timeit.timeit(
    stmt='np.cos(np.random.random())*np.random.random()',
    setup='import numpy as np')
print('Hard coded function 1 time: ', time_hardcoded1)

print('\nTest function 2: ')
time_sympy2 = timeit.timeit(
    stmt='f(np.random.random(), np.random.random(), np.random.random(),\
          np.random.random(), np.random.random(), np.random.random())',
    setup='import numpy as np;\
           from sympy.utilities.autowrap import autowrap;\
           import sympy as sp;\
           q0 = sp.Symbol("q0");\
           q1 = sp.Symbol("q1");\
           q2 = sp.Symbol("q2");\
           l0 = sp.Symbol("l0");\
           l1 = sp.Symbol("l1");\
           l2 = sp.Symbol("l2");\
           a = l1*sp.sin(q0 - l0*sp.sin(q1)*sp.cos(q2) - l2*sp.sin(q2) -\
           l0*sp.sin(q1) + q0*l0)*sp.cos(q0) + l2*sp.sin(q0);\
           f = autowrap(a, backend="cython", args=(q0, q1, q2, l0, l1, l2))')
print('Sympy autowrap function 2 time: ', time_sympy2)

time_hardcoded2 = timeit.timeit(
    stmt='q0 = np.random.random();\
          q1 = np.random.random();\
          q2 = np.random.random();\
          l0 = np.random.random();\
          l1 = np.random.random();\
          l2 = np.random.random();\
          l1*np.sin(q0 - l0*np.sin(q1)*np.cos(q2) - l2*np.sin(q2) -\
          l0*np.sin(q1) + q0*l0)*np.cos(q0) + l2*np.sin(q0)',
    setup='import numpy as np')
print('Hard coded function 2 time: ', time_hardcoded2)

plt.figure(figsize=(5, 3))
plt.bar([.5, 3.5], [time_sympy1, time_sympy2], color='blue', width=1)
plt.bar([1.5, 4.5], [time_hardcoded1, time_hardcoded2], color='red', width=1)
plt.xticks([1.25, 4.25], ['Function 1', 'Function 2'])
plt.ylabel('Simulation time (s)')
plt.xlim([.5, 5.5])
plt.legend(['Sympy with Cython', 'Hard-coded'], loc=2)
plt.tight_layout()
plt.show()
