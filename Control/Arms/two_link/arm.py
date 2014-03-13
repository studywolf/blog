'''
Copyright (C) 2013 Travis DeWolf

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
import py2LinkArm

class Arm2Link(Arm):
    """A wrapper around a MapleSim generated C simulation
    of a two link arm."""

    def __init__(self, **kwargs):

        Arm.__init__(self, singularity_thresh=1e-5, **kwargs)

        # length of arm links
        self.l1 = .31; self.l2 = .27
        self.L = np.array([self.l1, self.l2])
        # mass of links
        m1=1.98; m2=1.32
        # z axis inertia moment of links
        izz1=15.; izz2=8.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6)); self.M2 = np.zeros((6,6)) 
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[3:,3:] = np.eye(3)*izz1
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[3:,3:] = np.eye(3)*izz2

        self.rest_angles = np.array([np.pi/4.0, np.pi/4.0])
        
        # stores information returned from maplesim
        self.state = np.zeros(7) 
        # maplesim arm simulation
        self.sim = py2LinkArm.pySim(dt=self.dt)
        self.sim.reset(self.state)
        self.update_state()
 
    def apply_torque(self, u, dt):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly. 

        u np.array: the control signal to apply
        dt float: the timestep
        """
        u = np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/self.dt))):
            self.sim.step(self.state, u)
        self.update_state()

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

    def position(self, q=None, ee_only=False, rotate=0.0):
        """Compute x,y position of the hand

        q np.array: a set of angles to return positions for
        ee_only boolean: only return the (x,y) of the end-effector
        rotate float: how much to rotate the first joint by
        """
        if q is None: q0 = self.q[0]; q1 = self.q[1]
        else: q0 = q[0]; q1 = q[1]
        q0 += rotate

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)

    def update_state(self):
        """Separate out the state variable into time, angles, 
        velocities, and accelerations."""

        self.t = self.state[0]
        self.q = self.state[1:3]
        self.dq = self.state[3:5] 
        self.ddq = self.state[5:] 

        self.x = self.position(ee_only=True)

