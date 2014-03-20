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

from Arms.one_link.arm import Arm1Link as Arm1
from Arms.three_link.arm import Arm3Link as Arm3

import Controllers.osc as osc 
import Controllers.shell as shell

import numpy as np

def Task(arm_class, control_type):
    """
    This task sets up the arm to follow the mouse 
    with its end-effector.

    arm_class Arm: the arm class chosen for this task
    control_type Control: the controller class chosen for this task
    """

    if not issubclass(control_type, osc.Control):
        raise Exception('System must use operational space control '\
                        '(osc) for following mouse task.')

    if issubclass(arm_class, Arm1):
        raise Exception('System must can not use 1 link arm '\
                        'for following mouse task.')

    control_pars = {'pen_down':True}

    runner_pars = {'control_type':'osc',
                   'title':'Task: Follow mouse',
                   'mouse_control':True}
    if issubclass(arm_class, Arm3):
        runner_pars.update({'box':[-5,5,-5,5]})

    kp = 50 # position error gain on the PD controller
    controller = control_type(kp=kp, kv=np.sqrt(kp))
    control_shell = shell.Shell(controller=controller, **control_pars)

    return (control_shell, runner_pars)

