'''
Copyright (C) 2013 Terry Stewart & Travis DeWolf

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

from ..Arm import Arm
import numpy as np

class Arm2Link(Arm):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, dt=1e-5, l1=.31, l2=.27, **kwargs): 
        
        Arm.__init__(self, **kwargs)

        self.dt = dt # timestep 
        
        # length of arm links
        self.l1=l1; self.l2=l2
        self.L = np.array([self.l1, self.l2])
        # mass of links
        m1=1.98; m2=1.32
        # z axis inertia moment of links
        izz1=15; izz2=8
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6)); self.M2 = np.zeros((6,6)) 
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[3:,3:] = np.eye(3)*izz1
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[3:,3:] = np.eye(3)*izz2

        # compute non changing constants 
        self.K1 = (1/3.*m1+m2)*self.l1**2. + 1/3.*m2*self.l2**2.; 
        self.K2 = m2*self.l1*self.l2;
        self.K3 = 1/3.*m2*self.l2**2.; 
        self.K4 = 1/2.*m2*self.l1*self.l2; 
                    
        # initial arm joint and end-effector position
        self.q = np.array([0, np.pi/4.])
        self.x = self.position(ee_only=True)
        # initial arm joint and end-effector velocity
        self.dq = np.zeros(2)
        # initial arm joint and end-effector acceleration
        self.ddq = np.zeros(2)

        self.t = 0.0

    def apply_torque(self, u, dt=None):
        if dt is None: 
            dt = self.dt

        # equations solved for angles
        C2 = np.cos(self.q[1])
        S2 = np.sin(self.q[1])
        M11 = (self.K1 + self.K2*C2)
        M12 = (self.K3 + self.K4*C2)
        M21 = M12
        M22 = self.K3
        H1 = -self.K2*S2*self.dq[0]*self.dq[1] - 1/2.*self.K2*S2*self.dq[1]**2.
        H2 = 1/2.*self.K2*S2*self.dq[0]**2.

        self.ddq[1] = (H2*M11 - H1*M21 - M11*u[1]+ M21*u[0]) / (M12**2. - M11*M22)
        self.ddq[0] = (-H2 + u[1]- M22*self.ddq[1]) / M21
        self.dq[1] += self.ddq[1]*dt
        self.dq[0] += self.ddq[0]*dt
        self.q[0] += self.dq[0]*dt
        self.q[1] += self.dq[1]*dt
        self.x = self.position(ee_only=True)

        # transfer to next time step 
        self.t += dt

    def gen_jacCOM1(self):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
    
        JCOM1 = np.zeros((6,2))
        JCOM1[0,0] = self.l1 / 2. * -np.sin(self.q[0]) 
        JCOM1[1,0] = self.l1 / 2. * np.cos(self.q[0]) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self):
        """Generates the Jacobian from the COM of the second 
           link to the origin frame"""

        JCOM2 = np.zeros((6,2))
        # define column entries right to left
        JCOM2[0,1] = self.l2 / 2. * -np.sin(self.q[0] + self.q[1])
        JCOM2[1,1] = self.l2 / 2. * np.cos(self.q[0] + self.q[1])
        JCOM2[5,1] = 1.0

        JCOM2[0,0] = self.l1 * -np.sin(self.q[0]) + JCOM2[0,1]
        JCOM2[1,0] = self.l1 * np.cos(self.q[0]) + JCOM2[1,1]
        JCOM2[5,0] = 1.0

        return JCOM2

    def gen_jacEE(self):
        """Generates the Jacobian from end-effector to
           the origin frame"""

        JEE = np.zeros((2,2))
        # define column entries right to left
        JEE[0,1] = self.l2 * -np.sin(self.q[0] + self.q[1])
        JEE[1,1] = self.l2 * np.cos(self.q[0] + self.q[1]) 

        JEE[0,0] = self.l1 * -np.sin(self.q[0]) + JEE[0,1]
        JEE[1,0] = self.l1 * np.cos(self.q[0]) + JEE[1,1]
        
        return JEE

    def gen_Mq(self):
        """Generates the mass matrix for the arm in joint space"""
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1()
        JCOM2 = self.gen_jacCOM2()
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(self.M2, JCOM2))
        
        return Mq

    def position(self, q=None, ee_only=False):
        """Compute x,y position of the hand"""
        if q is None: q0 = self.q[0]; q1 = self.q[1]
        else: q0 = q[0]; q1 = q[1]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)
