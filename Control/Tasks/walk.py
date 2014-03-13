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

import Controllers.dmp as DMP
import Controllers.gc as GC

import csv
import numpy as np

def Task(arm_class, control_class):
    """
    This task sets up the arm to move like a leg walking.

    arm_class Arm: the arm class chosen for this task
    control_class Control: the controller class chosen for this task
    """

    if control_class not in (DMP, GC.Control):
        raise Exception('System must use generalized coordinates control '\
                        '(gc) or dynamic movement primitives (dmp) '\
                        'for walking task.')

    if not issubclass(arm_class, Arm3):
        raise Exception('System must use 3 link arm (arm3) for the '\
                        'walking task')

    #--------------------------------
    # set up the rhythmic trajectory that imitates leg walking
    # read in trajectories for each joint from their csv files
    dt = .01
    timesteps = int(1./dt)
    names = ['hip_traj.csv', 'knee_traj.csv', 'ankle_traj.csv']
    trajectory = np.zeros((timesteps+2, 3))*np.nan
    for ii, name in enumerate(names):
        with open('Tasks/Walk/'+name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            col = []
            for row in reader: 
                row = [float(val) for val in row]
                col.append(row[1])
            
            # generate function to interpolate the desired trajectory
            import scipy.interpolate
            path = np.zeros(timesteps)
            x = np.linspace(0, 1, len(col))
            path_gen = scipy.interpolate.interp1d(x, col)
            for t in range(timesteps):  
                path[t] = path_gen(t * dt)
            # we're only interested in the y-dimensions of each trajectory
            trajectory[1:-1, ii] = path

    # these trajectories are in degrees, we need to convert them into degrees
    trajectory *= (np.pi / 180.  )
    # also we need to sort them out for the arm 
    trajectory[:,1] *= -1 
    trajectory[:,2] += np.pi / 2.

    # number of goals is the number of (NANs - 1) * number of DMPs
    num_goals = (np.sum(trajectory[:,0] != trajectory[:,0]) - 1) * 3
    # respecify goals for spatial scaling by changing add_to_goals
    control_pars = {'add_to_goals':[1e-4]*num_goals,
                    'bfs':1000, # how many basis function per DMP
                    'gain':100, # pd gain for trajectory following
                    'pattern':'rhythmic', # type of DMP to use
                    'pen_down':False, 
                    'tau':.1, # tau is the time scaling term
                    'trajectory':trajectory.T,} 

    runner_pars = {'box':[-5,5,-5,5],
                   'control_type':'dmp',
                   'rotate':-np.pi/2.,
                   'title':'Task: Walking'}

    kp = 50 # position error gain on the PD controller
    controller = DMP.Control(base_class=GC.Control,
                             kp=kp, kv=np.sqrt(kp), **control_pars)

    return (controller, runner_pars)

