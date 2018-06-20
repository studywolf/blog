""" Copyright (C) 2018 Travis DeWolf

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
"""

import numpy as np
import matplotlib.pyplot as plt

def bootstrapci(data, func, n=3000, p=0.95):
    index = int(n*(1-p)/2)
    samples = np.random.choice(data, size=(n, len(data)))
    r = [func(s) for s in samples]
    r.sort()
    return r[index], r[-index]

fig = plt.figure(figsize=(7, 3.5))
for name, color in zip(
        ['policy_gradient', 'natural_policy_gradient'], ['b', 'g']):

    # load in data
    all_max_rewards = []
    all_total_episodes =[]
    for ii in range(10):
        data = np.load('data/%s_%i.npz' % (name, ii))
        all_max_rewards.append(data['max_rewards'])
        all_total_episodes.append(data['total_episodes'])
    all_max_rewards = np.array(all_max_rewards)
    all_total_episodes = np.array(all_total_episodes)

    # calculate mean
    mean = np.mean(all_max_rewards, axis=0)
    # calculate 95% confidence intervals
    sample = []
    upper_bound = []
    lower_bound = []
    for ii in range(all_max_rewards.shape[1]):
        data = all_max_rewards[:, ii]
        ci = bootstrapci(data, np.mean)
        sample.append(np.mean(data))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])

    plt.plot(
        range(all_max_rewards.shape[1]), mean, color=color, lw=2)
    plt.fill_between(
        range(all_max_rewards.shape[1]), upper_bound, lower_bound,
        color=color, alpha=.5)

plt.xlabel('Batch number')
plt.ylabel('Max reward from batch')
plt.legend(['Policy gradient', 'Natural policy gradient'], loc=4)
plt.tight_layout()
plt.show()
