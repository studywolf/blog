import numpy as np
import timeit

from abr_control.utils import transformations

q_A = transformations.random_quaternion()
trials = 10000

# check timing of transforming to Euler angles
times_transform = []
for ii in range(trials):
    start_time = timeit.default_timer()
    answer = transformations.euler_from_quaternion(q_A)
    times_transform.append(timeit.default_timer() - start_time)
avg_transform = np.sum(times_transform) / trials * 1000

# check timing of slice and sign multiply
times_slicesign = []
for ii in range(trials):
    start_time = timeit.default_timer()
    answer = q_A[1:] * np.sign(q_A[0])
    times_slicesign.append(timeit.default_timer() - start_time)
avg_slicesign = np.sum(times_slicesign) / trials * 1000

print('Average time of time of transform: ', avg_transform)
print('Average time of time of slice and sign multiply: ', avg_slicesign)

import matplotlib.pyplot as plt
plt.bar([.5, 1.5], [avg_transform, avg_slicesign])
plt.xticks([.5, 1.5], ['transform', 'slice and sign'])
plt.ylabel('Time (ms)')
plt.tight_layout()
plt.savefig('timing.png')
plt.show()
