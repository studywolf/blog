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

from Arms.three_link.arm import Arm3Link as Arm3

import numpy as np

def Task(arm_class, control_class):
    """
    This task sets up the arm to move to random 
    target positions ever t_target seconds. 
    """

    control_pars = {'pen_down':True}

    runner_pars = {'control_type':'random',
                   'title':'Task: Random movements'}
    if issubclass(arm_class, Arm3):
        runner_pars.update({'box':[-5,5,-5,5]})

    kp = 50 # position error gain on the PD controller
    controller = control_class(kp=kp, kv=np.sqrt(kp), **control_pars)

    return (controller, runner_pars)
