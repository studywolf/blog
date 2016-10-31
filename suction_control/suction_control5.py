""" An implementation based on the 2-link arm plant and controller from
    (Slotine & Sastry, 1983).
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

    def step(self, T):
        """ apply torques T and move one time step forward """

        Tprime_1 = (2*T[0] + np.sin(self.theta2[0])*self.theta2[1] *
                    (2*self.theta1[1] + self.theta2[1]))
        Tprime_2 = 2*T[1] - np.sin(self.theta2[0])*self.theta2[1]
        denom = (16.0/9.0 - np.cos(self.theta2[0])**2)

        ddtheta1 = (
            (2.0/3.0 * Tprime_1 - (2.0/3.0 +
                                   np.cos(self.theta2[0]) * Tprime_2)) / denom)
        ddtheta2 = (
            (-(2.0/3.0 + np.cos(self.theta2[0])) * Tprime_1 +
             2*(5.0/3.0 + np.cos(self.theta2[0])) * Tprime_2) / denom)

        self.theta1 += np.array([self.theta1[1], ddtheta1]) * self.dt
        self.theta2 += np.array([self.theta2[1], ddtheta2]) * self.dt

    @property
    def state(self):
        return self.theta1, self.theta2


class controller:
    def __init__(self):
        pass

    def control(self, t, theta1, theta2, theta1d, theta2d):
        """
        t float: the current time (desired trajectory is a function of time)
        thetas np.array: plant state
        """
        # calculate the value of s for generating our gains
        s1 = (theta1[0] - theta1d[0]) + 5*(theta1[1] - theta1d[1])
        s2 = (theta2[0] - theta2d[0]) + 5*(theta2[1] - theta2d[1])

        # gains for u1
        b11 = -0.7 if s1*theta2[1]*(2*theta1[1] + theta2[1]) > 0 else 0.7
        b12 = -1.2 if s1*(theta1[1]**2) > 0 else 1.2
        k11 = -9 if s1*(theta1[1] - theta1d[1]) > 0 else -3.8

        # gains for u2
        b21 = -1.2 if s2*theta2[1]*(2*theta1[1] + theta2[1]) > 0 else 1.2
        b22 = -4.4 if s2*(theta1[1]**2) > 0 else 4.4
        k21 = -9 if s2*(theta2[1] - theta2d[1]) > 0 else -3.8

        # shared gains
        k2 = 3.15

        u1 = (
            b11*theta2[1]*(2*theta1[1] + theta2[1]) + b12*theta1[1]**2 +
            k11*(theta1[1] - theta1d[1]) - k2*np.sign(s1))
        u2 = (
            b21*theta2[1]*(2*theta1[1] + theta2[1]) + b22*theta1[1]**2 +
            k21*(theta2[1] - theta2d[1]) - k2*np.sign(s2))

        T2 = ((u2 + (1 + 3.0/2.0*np.cos(theta2[0]))*u1) /
              (10.0/3.0 - np.cos(theta2[0])))
        T1 = 3.0/4.0*(u1 + (4.0/3.0 + 2*np.cos(theta2[0]))*T2)

        return np.array([T1, T2])

T = 5
dt = 0.001
timeline = np.arange(0.0, T, dt)

ctrlr = controller()

theta1 = [-90.0, 0.0]
theta2 = [170.0, 0.0]
plant_uncontrolled = plant(dt=dt, theta1=theta1, theta2=theta2)
theta1_uncontrolled_track = np.zeros((timeline.shape[0], 2))
theta2_uncontrolled_track = np.zeros((timeline.shape[0], 2))

plant_controlled = plant(dt=dt, theta1=theta1, theta2=theta2)
theta1_controlled_track = np.zeros((timeline.shape[0], 2))
theta2_controlled_track = np.zeros((timeline.shape[0], 2))

theta1d_1 = lambda t: -90 + 52.5*(1 - np.cos(1.26*t)) if t <= 2.5 else 15
theta1d_2 = lambda t: 52.5*1.26*np.sin(1.26*t) if t <= 2.5 else 0.0

theta2d_1 = lambda t: 170 - 60*(1 - np.cos(1.26*t)) if t <= 2.5 else 50
theta2d_2 = lambda t: -60*1.26*np.sin(1.26*t) if t < 2.5 else 0.0

thetad_track = np.zeros((timeline.shape[0], 2))

for ii, t in enumerate(timeline):

    if ii % int(1.0/dt) == 0:
        print('t: ', t)
    theta1d = [theta1d_1(t), theta1d_2(t)]
    theta2d = [theta2d_1(t), theta2d_2(t)]
    thetad_track[ii] = [theta1d[0], theta2d[0]]

    (theta1_uncontrolled_track[ii],
     theta2_uncontrolled_track[ii]) = plant_uncontrolled.state
    (theta1_controlled_track[ii],
     theta2_controlled_track[ii]) = plant_controlled.state

    u = ctrlr.control(t,
                      theta1_controlled_track[ii],
                      theta2_controlled_track[ii],
                      theta1d, theta2d)

    plant_uncontrolled.step(np.array([0.0, 0.0]))
    plant_controlled.step(u)

plt.subplot(2, 1, 1)
plt.plot(timeline, theta1_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta1_controlled_track[:, 0], lw=2)
plt.plot(timeline, thetad_track[:, 0], 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

plt.subplot(2, 1, 2)
plt.plot(timeline, theta2_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta2_controlled_track[:, 0], lw=2)
plt.plot(timeline, thetad_track[:, 1], 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

plt.tight_layout()
plt.show()
