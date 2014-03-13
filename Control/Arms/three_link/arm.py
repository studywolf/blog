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

import py3LinkArm 
import py3LinkArm_damping
import py3LinkArm_gravity
import py3LinkArm_gravity_damping 
import py3LinkArm_smallmass 

import numpy as np

class Arm3Link(Arm):
    """A wrapper around a MapleSim generated C simulation
    of a three link arm."""

    def __init__(self, **kwargs):

        Arm.__init__(self, **kwargs)

        pyArm = {None:py3LinkArm,
                    'damping':py3LinkArm_damping,
                    'gravity':py3LinkArm_gravity,
                    'gravity_damping':py3LinkArm_gravity_damping,
                    'smallmass':py3LinkArm_smallmass}[self.options]

        # length of arm links
        self.l1 = 2.0; self.l2 = 1.2; self.l3 = .7
        self.L = np.array([self.l1, self.l2, self.l3])
        # mass of links
        m1=10; m2=m1; m3=m1
        # z axis inertia moment of links
        izz1=100; izz2=izz1; izz3=izz1
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M2 = np.zeros((6,6)) 
        self.M3 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[5,5] = izz1
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[5,5] = izz2
        self.M3[0:3,0:3] = np.eye(3)*m3; self.M3[5,5] = izz3
        if self.options == 'smallmass': 
            self.M1 *= .001; self.M2 *= .001; self.M3 *= .001

        self.rest_angles = np.array([np.pi/4.0, np.pi/4.0, np.pi/4.0])
        
        # stores information returned from maplesim
        self.state = np.zeros(7) 
        # maplesim arm simulation
        self.sim = pyArm.pySim(dt=self.dt)
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
            self.sim.step(self.state, -1*u)
        self.update_state()

    def gen_jacCOM1(self):
        """Generates the Jacobian from the COM of the first
        link to the origin frame"""
        q0 = self.q[0]
    
        JCOM1 = np.zeros((6,3))
        JCOM1[0,0] = self.l1 / 2. * -np.sin(q0) 
        JCOM1[1,0] = self.l1 / 2. * np.cos(q0) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self):
        """Generates the Jacobian from the COM of the second 
        link to the origin frame"""
        q0 = self.q[0]
        q01 = self.q[0] + self.q[1]

        JCOM2 = np.zeros((6,3))
        # define column entries right to left
        JCOM2[0,1] = self.l2 / 2. * -np.sin(q01)
        JCOM2[1,1] = self.l2 / 2. * np.cos(q01)
        JCOM2[5,1] = 1.0

        JCOM2[0,0] = self.l1 * -np.sin(q0) + JCOM2[0,1]
        JCOM2[1,0] = self.l1 * np.cos(q0) + JCOM2[1,1]
        JCOM2[5,0] = 1.0

        return JCOM2
    
    def gen_jacCOM3(self): 
        """Generates the Jacobian from the COM of the third
        link to the origin frame"""

        q0 = self.q[0]
        q01 = self.q[0] + self.q[1] 
        q012 = self.q[0] + self.q[1] + self.q[2]

        JCOM3 = np.zeros((6,3))
        # define column entries right to left
        JCOM3[0,2] = self.l3 / 2. * -np.sin(q012)
        JCOM3[1,2] = self.l3 / 2. * np.cos(q012)
        JCOM3[5,2] = 1.0

        JCOM3[0,1] = self.l2 * -np.sin(q01) + JCOM3[0,2]
        JCOM3[1,1] = self.l2 * np.cos(q01) + JCOM3[1,2]
        JCOM3[5,1] = 1.0 

        JCOM3[0,0] = self.l1 * -np.sin(q0) + JCOM3[0,1]
        JCOM3[1,0] = self.l1 * np.cos(q0) + JCOM3[1,1]
        JCOM3[5,0] = 1.0 

        return JCOM3

    def gen_jacEE(self):
        """Generates the Jacobian from end-effector to
        the origin frame"""

        q0 = self.q[0]
        q01 = self.q[0] + self.q[1] 
        q012 = self.q[0] + self.q[1] + self.q[2]

        JEE = np.zeros((2,3))
        # define column entries right to left
        JEE[0,2] = self.l3 * -np.sin(q012)
        JEE[1,2] = self.l3 * np.cos(q012)

        JEE[0,1] = self.l2 * -np.sin(q01) + JEE[0,2]
        JEE[1,1] = self.l2 * np.cos(q01) + JEE[1,2]

        JEE[0,0] = self.l1 * -np.sin(q0) + JEE[0,1]
        JEE[1,0] = self.l1 * np.cos(q0) + JEE[1,1]

        return JEE

    def gen_djacEE(self):
        """Generates the Jacobian from end-effector to
        the origin frame"""

        q0 = self.q[0]
        dq0 = self.dq[0]
        q01 = self.q[0] + self.q[1] 
        dq01 = self.dq[0] + self.dq[1]
        q012 = self.q[0] + self.q[1] + self.q[2]
        dq012 = self.dq[0] + self.dq[1] + self.dq[2]

        dJEE = np.zeros((2,3))
        # define column entries right to left
        dJEE[0,2] = dq012 * self.l3 * -np.cos(q012)
        dJEE[1,2] = dq012 * self.l3 * -np.sin(q012)

        dJEE[0,1] = dq01 * self.l2 * -np.cos(q01) + dJEE[0,2]
        dJEE[1,1] = dq01 * self.l2 * -np.sin(q01) + dJEE[1,2]

        dJEE[0,0] = dq0 * self.l1 * -np.cos(q0) + dJEE[0,1]
        dJEE[1,0] = dq0 * self.l1 * -np.sin(q0) + dJEE[1,1]

        return dJEE

    def gen_Mq(self):
        """Generates the mass matrix of the arm in joint space"""

        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1()
        JCOM2 = self.gen_jacCOM2()
        JCOM3 = self.gen_jacCOM3()
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(self.M2, JCOM2)) + \
             np.dot(JCOM3.T, np.dot(self.M3, JCOM3))

        return Mq

    def position(self, q=None, ee_only=False, rotate=0.0):
        """Compute x,y position of the hand

        q np.array: a set of angles to return positions for
        ee_only boolean: only return the (x,y) of the end-effector
        rotate float: how much to rotate the first joint by
        """
        if q is None: q0 = self.q[0]; q1 = self.q[1]; q2 = self.q[2]
        else: q0 = q[0]; q1 = q[1]; q2 = q[2]
        q0 += rotate

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1), 
                       self.l3 * np.cos(q0+q1+q2)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1),
                       self.l3 * np.sin(q0+q1+q2)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)

    def update_state(self):
        """Update the local variables"""
        self.t = self.state[0]
        self.q = self.state[1:4]
        self.dq = self.state[4:] 

        self.x = self.position(ee_only=True)

