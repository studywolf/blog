'''
Copyright (C) 2015 Travis DeWolf

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
import numpy as np

class Arm2Link:
    """A base class for arm simulators"""

    def __init__(self, dt=1e-4):
        """
        dt float: the timestep for simulation
        singularity_thresh float: the point at which to singular values
                                  from the matrix SVD to zero.
        """
        self.DOF = 2
        self.dt = dt

        # length of arm links
        self.l1 = 1.5
        self.l2 = 1.3

        # mass of links
        m1=1.98
        m2=1.32

        # compute non changing constants 
        self.K1 = (1/3.*m1+m2)*self.l1**2. + 1/3.*m2*self.l2**2.; 
        self.K2 = m2*self.l1*self.l2;
        self.K3 = 1/3.*m2*self.l2**2.; 
        self.K4 = 1/2.*m2*self.l1*self.l2; 
 
        self.reset()

    def apply_torque(self, u, dt=None):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly. 

        u np.array: the control signal to apply
        dt float: the timestep
        """
        if dt is None: 
            dt = self.dt

        # equations solved for angles
        C2 = np.cos(self.q1)
        S2 = np.sin(self.q1)
        M11 = (self.K1 + self.K2*C2)
        M12 = (self.K3 + self.K4*C2)
        M21 = M12
        M22 = self.K3
        H1 = -self.K2*S2*self.dq0*self.dq1 - 1/2.*self.K2*S2*self.dq1**2.
        H2 = 1/2.*self.K2*S2*self.dq0**2.

        ddq1 = (H2*M11 - H1*M21 - M11*u[1]+ M21*u[0]) / (M12**2. - M11*M22)
        ddq0 = (-H2 + u[1]- M22*ddq1) / M21
        self.dq1 = self.dq1 + ddq1*dt
        self.dq0 = self.dq0 + ddq0*dt
        self.q1 = self.q1 + self.dq1*dt
        self.q0 = self.q0 + self.dq0*dt
        self.x = self.position(ee_only=True)

    def position(self, q=None, ee_only=False):
        """Compute x,y position of the hand

        q list: a list of the joint angles, 
                if None use current system state
        ee_only boolean: if true only return the 
                         position of the end-effector
        """
        q0 = self.q0 if q is None else q[0]
        q1 = self.q1 if q is None else q[1]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1)])

        if ee_only: 
            return np.array([x[-1], y[-1]])

        return (x, y)

    def reset(self, q=[], dq=[]):
        """Resets the state of the arm 
        q list: a list of the joint angles
        dq list: a list of the joint velocities
        """
        if isinstance(q, np.ndarray): q = q.tolist()
        if isinstance(dq, np.ndarray): dq = dq.tolist()

        self.q0 = np.pi/2; self.q1 = np.pi/2
        self.dq0 = 0; self.dq1 = 0
        self.x = self.position(ee_only=True)
