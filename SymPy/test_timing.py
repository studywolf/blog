'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import timeit

# Test 1 ----------------------------------------------------------------------
print('\nTest function 1: ')
time_sympy1 = timeit.timeit(
        stmt = 'f(np.random.random(), np.random.random())', 
       setup = 'import numpy as np;\
                import sympy as sp;\
                q0 = sp.Symbol("q0");\
                l0 = sp.Symbol("l0");\
                a = sp.cos(q0) * l0;\
                f = sp.lambdify((q0, l0), a, "numpy")')
print('Sympy lambdify function 1 time: ', time_sympy1)

time_hardcoded1 = timeit.timeit(
        stmt = 'np.cos(np.random.random())*np.random.random()', 
        setup = 'import numpy as np')
print('Hard coded function 1 time: ', time_hardcoded1)

# Test 2 ----------------------------------------------------------------------
print('\nTest function 2: ')
time_sympy2 = timeit.timeit(
        stmt = 'f(np.random.random(), np.random.random(), np.random.random(),\
                np.random.random(), np.random.random(), np.random.random())',
        setup = 'import numpy as np;\
                import sympy as sp;\
                q0 = sp.Symbol("q0");\
                q1 = sp.Symbol("q1");\
                q2 = sp.Symbol("q2");\
                l0 = sp.Symbol("l0");\
                l1 = sp.Symbol("l1");\
                l2 = sp.Symbol("l2");\
                a = l1*sp.sin(q0 - l0*sp.sin(q1)*sp.cos(q2) - l2*sp.sin(q2) - l0*sp.sin(q1) + q0*l0)*sp.cos(q0) + l2*sp.sin(q0);\
                f = sp.lambdify((q0,q1,q2,l0,l1,l2), a, "numpy")')
print('Sympy lambdify function 2 time: ', time_sympy2)

time_hardcoded2 = timeit.timeit(
        stmt = 'l1*np.sin(q0 - l0*np.sin(q1)*np.cos(q2) - l2*np.sin(q2) - l0*np.sin(q1) + q0*l0)*np.cos(q0) + l2*np.sin(q0)',
        setup = 'import numpy as np;\
                q0 = np.random.random();\
                q1 = np.random.random();\
                q2 = np.random.random();\
                l0 = np.random.random();\
                l1 = np.random.random();\
                l2 = np.random.random()')
print('Hard coded function 2 time: ', time_hardcoded2)

import numpy as np
ind = np.arange(2)
width = 0.35
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,3))
plt.bar(ind, [time_sympy1, time_sympy2], width, color='b')
plt.bar(ind + width, [time_hardcoded1, time_hardcoded2], width, color='r')
plt.xlim([0, ind[-1]+width*2])
plt.ylabel('Simulation time (s)')
ax.set_xticks(ind + width)
ax.set_xticklabels(['Function 1', 'Function 2'])
ax.legend(['Sympy', 'Hard-coded'], loc='best')

plt.show()
