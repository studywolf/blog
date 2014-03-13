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

def Control(base_class, dt, gain, trajectory, 
                threshold=.01, **kwargs):
    """
    A controller that uses a given trajectory to 
    control a robotic arm end-effector.

    base_class Class: the class to inherit control from
    dt float: the timestep with which to move along the given trajectory
    gain float: the PD gain while following a DMP trajectory
    trajectory np.array: the time series of points to follow
                         [DOFs, time], with a column of None
                         wherever the pen should be lifted
    """

    control_class = type('Control_Trajectory', (base_class,), 
                         {'base_class':base_class,
                          'control':control,
                          'done':False,
                          'dt':dt,
                          'gain':gain,
                          'gen_trajectories':gen_trajectories,
                          'not_at_start':True,
                          'num_seq':0,
                          'threshold':threshold,
                          'time':0.0})
    controller = control_class(**kwargs)

    controller.gen_trajectories(trajectory)
    controller.target = np.array([controller.trajectory[0](0.0), 
                                  controller.trajectory[1](0.0)])

    return controller

def control(self, arm):
    """Drive the end-effector through a series 
       of (x,y) points"""

    if self.check_distance(arm) < self.threshold:
        self.not_at_start = False

    if self.not_at_start or self.done:
        u = self.base_class.control(self, arm)

    else: 
        y = np.array([self.trajectory[d](self.time) for d in range(2)])
        self.time += self.dt

        # check to see if it's pen up time
        if self.time >= 1: 
            self.pen_down = False
            self.time = 0.0
            if self.num_seq >= len(self.seqs_x) - 1:
                # if we're finished the last sequence
                self.done = True
                self.target = [.3, 0]
            else: 
                # else move on to the next sequence
                self.not_at_start = True
                self.num_seq += 1
                self.trajectory = [self.seqs_x[self.num_seq], 
                              self.seqs_y[self.num_seq]]
                self.target = [self.trajectory[0](0.0), self.trajectory[1](0.0)]
        else: 
            self.pen_down = True
        
        self.x = arm.position(ee_only=True)
        x_des = self.gain * (y - self.x)
        u = self.base_class.control(self, arm, x_des=x_des)

    return u

def gen_trajectories(self, trajectory):
    """Generates the trajectories for the 
    position, velocity, and acceleration to follow
    during run time to reproduce the given trajectory.

    trajectory np.array: a list of points to follow
    """

    if trajectory.ndim == 1: 
        trajectory = trajectory.reshape(1,len(trajectory))
    dt = 1.0 / trajectory.shape[1]

    # break up the trajectory into its different words
    # NaN or None signals a new word / break in drawing
    breaks = np.where(trajectory != trajectory)
    # some vector manipulation to get what we want
    breaks = breaks[1][:len(breaks[1])/2]
   
    import scipy.interpolate
    self.seqs_x = [] 
    self.seqs_y = [] 
    for ii in range(len(breaks) - 1):
        # get the ii'th sequence
        seq_x = trajectory[0, breaks[ii]+1:breaks[ii+1]]
        seq_y = trajectory[1, breaks[ii]+1:breaks[ii+1]]
        
        # generate function to interpolate the desired trajectory
        vals = np.linspace(0, 1, len(seq_x))
        self.seqs_x.append(scipy.interpolate.interp1d(vals, seq_x))
        self.seqs_y.append(scipy.interpolate.interp1d(vals, seq_y))

    self.trajectory = [self.seqs_x[0], self.seqs_y[0]]

