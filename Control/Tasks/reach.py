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
import Controllers.gc as gc
import Controllers.osc as osc 
import Controllers.trace as trace
import Controllers.target_list as target_list

import numpy as np

def Task(arm_class, control_type, x_bias=0., y_bias=.25, dist=.15):
    """
    This task sets up the arm to reach to 8 targets center out from
    (x_bias, y_bias) at a distance=dist.
    """

    if issubclass(arm_class, Arm1):
        raise Exception('System must can not use 1 link arm '\
                        'for the reaching task')
    if issubclass(arm_class, Arm3):
        y_bias = 2.; dist = .4

    # set up the reaching trajectories, 8 points around unit circle
    repeat = 15
    targets_x = [dist * np.cos(theta) + x_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    targets_x += targets_x * repeat
    targets_y = [dist * np.sin(theta) + y_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    targets_y += targets_y * repeat
    trajectory = np.ones((3*len(targets_x)+1, 2))*np.nan

    for ii in range(len(targets_x)): 
        trajectory[ii*3+1] = [0, y_bias]
        trajectory[ii*3+2] = [targets_x[ii], targets_y[ii]]
    
    kp = 10 # position gain on PD control
    kv = np.sqrt(kp) # velocity gain on PD control

    controller = osc.Control(kp=kp, 
                             kv=kp, 
                             null_control=True)

    # TODO: make it possible to reach with trace 
    if issubclass(control_type, dmp.Shell):

        # number of goals is the number of (NANs - 1) * number of DMPs
        num_goals = (np.sum(trajectory[:,0] != trajectory[:,0]) - 1) * 3
        # respecify goals for spatial scaling by changing add_to_goals
        control_pars = {'add_to_goals':[1e-4]*num_goals,
                        'bfs':1000, # how many basis function per DMP
                        'gain':1000, # pd gain for trajectory following
                        'tau':.01, # tau is the time scaling term
                        'trajectory':trajectory.T} 

        runner_pars = {'control_type':'dmp'}


    elif issubclass(control_type, osc.Control):

        control_pars = {'target_list':trajectory}
        runner_pars = {'control_type':'osc'}

        control_type = target_list.Shell

    else:
        raise Exception('System must use DMP, or OSC control'\
                        '(dmp | osc) for reaching tasks.')

    control_pars.update({'threshold':.015}) # how close to get to each target
    runner_pars.update({'infinite_trail':True, 
                        'title':'Task: Reaching',
                        'trajectory':trajectory})

    if issubclass(arm_class, Arm3):
        runner_pars.update({'box':[-5,5,-5,5]})

    control_shell = control_type(controller=controller, **control_pars)

    return (control_shell, runner_pars)
