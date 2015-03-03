'''
Copyright (C) 2014 Travis DeWolf

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

class Shell(object):
    """
    """

    def __init__(self, controller, target_list, 
                 threshold=.01, pen_down=False):
        """
        control Control instance: the controller to use 
        pen_down boolean: True if the end-effector is drawing
        """

        self.controller = controller
        self.pen_down = pen_down 
        self.target_list = target_list
        self.threshold = threshold
   
        self.not_at_start = True 
        self.target_index = 0
        self.set_target()

    def control(self, arm): 
        """Move to a series of targets.
        """

        if self.controller.check_distance(arm) < self.threshold:
            if self.target_index < len(self.target_list)-1:
                self.target_index += 1
            self.set_target()

            self.controller.apply_noise = True
            self.not_at_start = not self.not_at_start
            self.pen_down = not self.pen_down

        self.u = self.controller.control(arm)

        return self.u

    def set_target(self):
        """
        Set the current target for the controller.
        """
        if self.target_index == len(self.target_list)-1:
            target = [1, 2]
        else:
            target = self.target_list[self.target_index]

        if target[0] != target[0]: # if it's NANs
            self.target_index += 1
            self.set_target()
        else:
            self.controller.target = target
