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

import Controllers.dmp as DMP
import Controllers.gc as GC 
import Controllers.osc as OSC

import numpy as np

def Task(arm_class, control_class, x_bias=0., y_bias=.25, dist=.15):
    """
    This task sets up the arm to reach to 8 targets center out from
    (x_bias, y_bias) at a distance=dist.
    """

    if control_class == GC.Control:
        raise Exception('System must use operational space control '\
                        '(osc) for reaching task.')

    if issubclass(arm_class, Arm1):
        raise Exception('System must can not use 1 link arm '\
                        'for the reaching task')
    if issubclass(arm_class, Arm3):
        y_bias = 2.; dist = 1.

    # set up the reaching trajectories, 8 points around unit circle
    targets_x = [dist * np.cos(theta) + x_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    targets_y = [dist * np.sin(theta) + y_bias \
                    for theta in np.linspace(0, np.pi*2, 9)][:-1]
    trajectory = np.ones((3*len(targets_x)+1, 2))*np.nan

    for ii in range(len(targets_x)): 
        trajectory[ii*3+1] = [0, y_bias]
        trajectory[ii*3+2] = [targets_x[ii], targets_y[ii]]

    # number of goals is the number of (NANs - 1) * number of DMPs
    num_goals = (np.sum(trajectory[:,0] != trajectory[:,0]) - 1) * 3
    # respecify goals for spatial scaling by changing add_to_goals
    control_pars = {'add_to_goals':[1e-4]*num_goals,
                    'bfs':1000, # how many basis function per DMP
                    'gain':1000, # pd gain for trajectory following
                    'pen_down':False, 
                    'tau':.01, # tau is the time scaling term
                    'trajectory':trajectory.T} 

    runner_pars = {'control_type':'dmp',
                   'infinite_trail':True, 
                   'title':'Task: Reaching',
                   'trajectory':trajectory}
    if issubclass(arm_class, Arm3):
        control_pars.update({'threshold':.1})
        runner_pars.update({'box':[-5,5,-5,5]})

    kp = 50 # position error gain on the PD controller
    controller = DMP.Control(base_class=OSC.Control,
                             kp=kp, kv=np.sqrt(kp), **control_pars)

    return (controller, runner_pars)
