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

from control import Control
import numpy as np

class Control_GC(Control):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, **kwargs): 

        Control.__init__(self, **kwargs)

        self.target = None

        # generalized coordinates
        self.target_gain = 2*np.pi
        self.target_bias = -np.pi

    def gen_target(self, arm):
        """Generate a random target"""
        self.target = np.random.random(size=(len(arm.L),)) * \
            self.target_gain + self.target_bias

        return arm.position(self.target)

    def control(self, arm):
        """Generate a control signal to move the arm through
           joint space to the desired joint angle position"""
        
        # calculated desired joint angle acceleration
        prop_val = ((self.target - arm.q) + np.pi) % \
                                                    (np.pi*2) - np.pi
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq).reshape(-1,)

        Mq = arm.gen_Mq()

        # tau = Mq * q_des + tau_grav, but gravity = 0
        self.u = np.dot(Mq, q_des).reshape(-1,)

        return self.u
