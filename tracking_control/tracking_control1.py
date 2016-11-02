""" An implementation of the plant and controller from example 1 of
    (Slotine & Sastry, 1983)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn


class plant:
    def __init__(self, dt=.001, theta1=[0.0, 0.0], theta2=[0.0, 0.0]):
        """
        dt float: simulation time step
        theta1 list/np.array: [position, velocity]
        theta2 list/np.array: [position, velocity]
        """
        self.dt = dt
        self.theta1 = np.asarray(theta1)
        self.theta2 = np.asarray(theta2)

    def step(self, u):
        """ apply u and move one time step forward """
        ddtheta1 = (
            3*self.theta1[0] + self.theta2[0]**2 +
            2*self.theta1[0] * self.theta2[1] * np.cos(self.theta2[0]) + u[0])
        ddtheta2 = (
            self.theta1[0]**3 - np.cos(self.theta1[0]) * self.theta2[0] + u[1])

        self.theta1 += np.array([self.theta1[1], ddtheta1]) * self.dt
        self.theta2 += np.array([self.theta2[1], ddtheta2]) * self.dt

    @property
    def state(self):
        return self.theta1, self.theta2


class controller:
    def __init__(self):
        pass

    def control(self, t, theta1, theta2):
        """
        t float: the current time (desired trajectory is a function of time)
        thetas np.array: plant state
        """
        # calculate the value of s for generating our gains
        s1 = theta1[1] + theta1[0] - 2*t*(t + 2)
        s2 = theta2[1] + theta2[0] - t*(t + 2)

        # gains for u1
        beta11 = -3
        beta12 = -1
        beta13 = -2 if s1 * theta1[0] * theta2[1] > 0 else 2
        k11 = -1
        k12 = 5  # k12 > 4

        # gains for u2
        beta21 = -1
        beta22 = -1 if s2 * theta2[0] > 0 else 1
        k21 = -1
        k22 = 3  # k22 > 2

        u1 = (
            beta11*theta1[0] + beta12*theta2[0]**2 +
            beta13*theta1[0]*theta2[1] + k11*(theta1[1] - 4*t) -
            k12*np.sign(s1))
        u2 = (
            beta21*theta1[0]**3 + beta22*theta2[0] +
            k21*(theta2[1] - 2*t) - k22*np.sign(s2))

        return np.array([u1, u2])


T = 1.0
dt = 0.001
timeline = np.arange(0.0, T, dt)

ctrlr = controller()

plant_uncontrolled = plant(dt=dt)
theta1_uncontrolled_track = np.zeros((timeline.shape[0], 2))
theta2_uncontrolled_track = np.zeros((timeline.shape[0], 2))

plant_controlled = plant(dt=dt)
theta1_controlled_track = np.zeros((timeline.shape[0], 2))
theta2_controlled_track = np.zeros((timeline.shape[0], 2))

for ii, t in enumerate(timeline):

    if ii % int(1.0/dt) == 0:
        print('t: ', t)
    (theta1_uncontrolled_track[ii],
     theta2_uncontrolled_track[ii]) = plant_uncontrolled.state
    (theta1_controlled_track[ii],
     theta2_controlled_track[ii]) = plant_controlled.state

    u = ctrlr.control(t,
                      theta1_controlled_track[ii],
                      theta2_controlled_track[ii])

    plant_uncontrolled.step(np.array([0.0, 0.0]))
    plant_controlled.step(u)

plt.subplot(2, 1, 1)
plt.plot(timeline, theta1_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta1_controlled_track[:, 0], lw=2)
plt.plot(timeline, 2*timeline**2, 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

plt.subplot(2, 1, 2)
plt.plot(timeline, theta2_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta2_controlled_track[:, 0], lw=2)
plt.plot(timeline, timeline**2, 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

plt.tight_layout()
plt.show()
