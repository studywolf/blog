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

from .DMPs import dmp as DMP
from .DMPs import dmp_discrete as DMP_discrete
from .DMPs import dmp_rhythmic as DMP_rhythmic
import gc as GC
import osc as OSC

import numpy as np

def Control(base_class, bfs, gain, trajectory, tau,
                add_to_goals=None, pattern='discrete',
                threshold=.01, **kwargs):
    """
    A controller that uses dynamic movement primitives to 
    control a robotic arm end-effector.

    base_class Class: the class to inherit control from
    bfs int: the number of basis functions per DMP
    gain float: the PD gain while following a DMP trajectory
    trajectory np.array: the time series of points to follow
                         [DOFs, time], with a column of None
                         wherever the pen should be lifted
    tau float: the time scaling term
    add_to_goals np.array: floats to add to the DMP goals
                           used to scale the DMPs spatially
    pattern string: specifies either 'discrete' or 'rhythmic' DMPs
    """

    control_class = type('Control_DMP', (base_class,), 
                         {'base_class':base_class,
                          'control':control,
                          'done':False,
                          'gain':gain,
                          'gen_dmps':gen_dmps,
                          'not_at_start':True,
                          'num_seq':0,
                          'tau':tau,
                          'threshold':threshold})
    controller = control_class(**kwargs)

    controller.gen_dmps(trajectory, bfs, pattern)
    controller.target,_,_ = controller.dmps.step(tau=controller.tau)

    if add_to_goals is not None: 
        for ii, dmp in enumerate(controller.dmp_sets):
            dmp.goal[0] += add_to_goals[ii*2]
            dmp.goal[1] += add_to_goals[ii*2+1]

    return controller

def control(self, arm): 
    """Apply a given control signal in (x,y) 
       space to the arm"""
        
    if self.check_distance(arm) < .01:
        self.not_at_start = False

    if self.not_at_start or self.done:
        u = self.base_class.control(self, arm)

    else:
        y,_,_ = self.dmps.step(tau=self.tau)#, state_fb=self.x)

        # check to see if it's pen up time
        if self.dmps.cs.x < \
            np.exp(-self.dmps.cs.ax * self.dmps.cs.run_time):

                self.pen_down = False
                
                if self.num_seq >= len(self.dmp_sets) - 1:
                    # if we're finished the last DMP
                    self.done = True
                    self.target = [.3, 0]
                else:
                    # else move on to the next DMP
                    self.not_at_start = True
                    self.num_seq += 1
                    self.dmps = self.dmp_sets[self.num_seq]
                    self.target,_,_ = self.dmps.step(tau=self.tau)
        else:
            self.pen_down = True

        if issubclass(self.base_class, OSC.Control):
            pos = arm.position(ee_only=True)
        elif issubclass(self.base_class, GC.Control):
            pos = arm.q

        pos_des = self.gain * (y - pos)
        u = self.base_class.control(self, arm, pos_des) 

    return u

def gen_dmps(self, trajectory, bfs, pattern):
    """Generate the DMPs necessary to follow the 
    specified trajectory.

    trajectory np.array: the time series of points to follow
                         [DOFs, time], with a column of None
                         wherever the pen should be lifted
    """

    if trajectory.ndim == 1: 
        trajectory = trajectory.reshape(1,len(trajectory))

    num_DOF = trajectory.shape[0]
    # break up the trajectory into its different words
    # NaN or None signals a new word / break in drawing
    breaks = np.array(np.where(trajectory[0] != trajectory[0]))[0] 

    self.dmp_sets = []
    for ii in range(len(breaks) - 1):
        # get the ii'th sequence
        seq = trajectory[:, breaks[ii]+1:breaks[ii+1]]

        if pattern == 'discrete':
            dmps = DMP_discrete.DMPs_discrete(dmps=num_DOF, bfs=bfs)
        elif pattern == 'rhythmic': 
            dmps = DMP_rhythmic.DMPs_rhythmic(dmps=num_DOF, bfs=bfs)
        else: 
            raise Exception('Invalid pattern type specified. Valid choices \
                             are discrete or rhythmic.')

        dmps.imitate_path(y_des=seq)
        self.dmp_sets.append(dmps)
        self.target,_,_ = dmps.step(tau=.002)

    self.dmps = self.dmp_sets[0]
