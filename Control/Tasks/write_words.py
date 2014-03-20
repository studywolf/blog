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

import Controllers.dmp as dmp
import Controllers.osc as osc 
import Controllers.trace as trace
import Controllers.trajectory as trajectory_class

import Tasks.Write.read_trajectory as rt

import numpy as np

# TODO: subclass trajectory tracing tasks
def Task(arm_class, control_type, 
            writebox=np.array([-.35,.35,.25,.4])):
    """
    This task sets up the arm to write numbers inside 
    a specified area (-x_bias, x_bias, -y_bias, y_bias). 
    """

    if not issubclass(control_type, trajectory_class.Shell):
        raise Exception('System must use trajectory control'\
                        '(dmp | trace) for writing tasks.')
    
    if issubclass(arm_class, Arm1):
        raise Exception('System must can not use 1 link arm '\
                        'for writing tasks')
    if issubclass(arm_class, Arm3):
        writebox=np.array([-2., 2., 1., 2.])

    #TODO: handle these two different cases better than commenting one out
    trajectory = rt.read_file('Tasks/Write/ca0.dat', '', box=writebox)
    #trajectory = rt.read_file_pkl('Tasks/Write/pen_pos.pkl',
                                  #'', box=writebox)[:210]
    #trajectory[-1] = np.array([np.nan, np.nan])

    control_pars = {'gain':1000, # pd gain for trajectory following
                    'pen_down':False, 
                    'trajectory':trajectory.T} 

    if issubclass(control_type, dmp.Shell):
        # number of goals is the number of (NANs - 1) * number of DMPs
        num_goals = (np.sum(trajectory[:,0] != trajectory[:,0]) - 1) * 2
        # respecify goals for spatial scaling by changing add_to_goals
        control_pars.update({'add_to_goals':[0]*num_goals,
                             'bfs':1000, # how many basis function per DMP
                             'tau':.005}) # tau is the time scaling term
    else:
        # trajectory based control
        control_pars.update({'tau':.00005}) # how fast the trajectory rolls out

    runner_pars = {'control_type':'write_words',
                   'infinite_trail':True, 
                   'title':'Task: Writing numbers',
                   'trajectory':trajectory}
    if issubclass(arm_class, Arm3):
        control_pars.update({'threshold':.1})
        runner_pars.update({'box':[-5,5,-5,5]})

    kp = 50 # position error gain on the PD controller
    controller = osc.Control(kp=kp, kv=np.sqrt(kp))
    controller = control_type(controller=controller, **control_pars) 
    
    return (controller, runner_pars)
