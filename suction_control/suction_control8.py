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

        mu = 0  # load on the arm
        ddtheta2 = (
            (2*T[1] - np.sin(theta2[0])*theta1[1]**2*(1 + 2*mu)) /
            ((4.0/3.0 + np.cos(theta2[0]) + 2*mu*(1 + np.cos(theta2[0])) +
             2*mu)))
        ddtheta1 = (
            (2*T[0] + np.sin(theta1[0])*theta2[1]**2*(2*theta1[1] + theta2[1]) *
             (1 + 2*mu) - ddtheta2*(2.0/3.0 + np.cos(theta2[0]) + 2*mu*(1 +
                                   np.cos(theta2[0])))) /
             (2*(5.0/3.0 + np.cos(theta2[0])) + 4*mu*(1 + np.cos(theta2[0]))))

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
        b11 = -1.2 if s1*theta2[1]*(2*theta1[1] + theta2[1]) > 0 else 1.2
        b12 = -2.1 if s1*(theta1[1]**2) > 0 else 2.1
        k11 = -15.2 if s1*(theta1[1] - theta1d[1]) > 0 else -2.4

        # gains for u2
        b21 = -2.1 if s2*theta2[1]*(2*theta1[1] + theta2[1]) > 0 else 2.1
        b22 = -6.4 if s2*(theta1[1]**2) > 0 else 6.4
        k21 = -15.2 if s2*(theta2[1] - theta2d[1]) > 0 else -2.4

        # shared gains
        k2 = 10.0

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

T = 30
dt = 0.001
timeline = np.arange(0.0, T, dt)

ctrlr = controller()

theta1 = [-50.0, 0.0]
theta2 = [100.0, 0.0]
plant_uncontrolled = plant(dt=dt, theta1=theta1, theta2=theta2)
theta1_uncontrolled_track = np.zeros((timeline.shape[0], 2))
theta2_uncontrolled_track = np.zeros((timeline.shape[0], 2))

plant_controlled = plant(dt=dt, theta1=theta1, theta2=theta2)
theta1_controlled_track = np.zeros((timeline.shape[0], 2))
theta2_controlled_track = np.zeros((timeline.shape[0], 2))

theta1d_1 = lambda t: -50 + 52.5*(1 - np.cos(1.26*t)) if t <= 30 else 50
theta1d_2 = lambda t: 52.5*1.26*np.sin(1.26*t) if t <= 30 else 0.0

theta2d_1 = lambda t: 170 - 60*(1 - np.sin(1.26*t)) if t <= 30 else 170
theta2d_2 = lambda t: 60*1.26*np.cos(1.26*t) if t < 30 else 0.0

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

plt.figure(figsize=(4, 8))
plt.subplot(4, 1, 1)
plt.plot(timeline, theta1_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta1_controlled_track[:, 0], lw=2)
plt.plot(timeline, thetad_track[:, 0], 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

plt.subplot(4, 1, 2)
plt.plot(timeline, theta2_uncontrolled_track[:, 0], lw=2)
plt.plot(timeline, theta2_controlled_track[:, 0], lw=2)
plt.plot(timeline, thetad_track[:, 1], 'r--', lw=2)
plt.legend(['uncontrolled', 'controlled', 'target'])

ax = plt.subplot(2, 1, 2)
ax.set_aspect('equal')
print(theta1_controlled_track[:, 0].shape)
def kin(q0, q1):
    # convert to hand (x,y) coordinates
    # L0 = L1 = 1
    q0 *= np.pi / 180.0
    q1 *= np.pi / 180.0
    return np.cos(q0) + np.cos(q0+q1), np.sin(q0) + np.sin(q0+q1)
x,y = kin(theta1_controlled_track[:, 0], theta2_controlled_track[:, 0])
xd, yd = kin(thetad_track[:, 0], thetad_track[:, 1])
plt.plot(x, y)
plt.plot(xd, yd, 'r--', lw=2)

plt.tight_layout()
plt.show()
