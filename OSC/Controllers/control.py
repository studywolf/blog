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

import numpy as np

class Control:
    """
    The base class for controllers.
    """
    def __init__(self, kp=10, kv=np.sqrt(10), pen_down=True):
        """
        kp float: the position error term gain value
        kv float: the velocity error term gain value
        pen_down boolean: True if the end-effector is drawing
        """

        self.pen_down = pen_down
        self.u = np.zeros((2,1)) # control signal

        self.kp = kp
        self.kv = kv
       
        np.random.seed(1)

    def control(self): 
        """Generates a control signal to apply to the arm"""
        raise NotImplementedError

    def gen_target(self, arm):
        """Generate a target based on the control type
        
        arm Arm: an instance of the arm object
        """
        raise NotImplementedError

    def set_target_from_mouse(self, target):
        """Takes in a (x,y) coordinate and sets the target
        
        target np.array: the (x,y) target location
        """
        self.target = target
        return self.target

