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

from control_OSC import Control_OSC
import numpy as np

class Control_trajectory(Control_OSC):
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, dt, gain, trajectory, **kwargs): 

        Control_OSC.__init__(self, **kwargs)

        self.dt = dt
        self.gain = gain
        self.gen_trajectories(trajectory)
        self.target = np.array([self.y_des[0](0.0), self.y_des[1](0.0)])

        self.num_seq = 0
        self.time = 0.0

        self.done = False
        self.not_at_start = True

    def gen_trajectories(self, y_des):
        """Generates the trajectories for the 
        position, velocity, and acceleration to follow
        during run time to reproduce the given trajectory.

        trajectory np.array: a list of points to follow
        """

        if y_des.ndim == 1: 
            y_des = y_des.reshape(1,len(y_des))
        dt = 1.0 / y_des.shape[1]

        # break up the trajectory into its different words
        # NaN or None signals a new word / break in drawing
        breaks = np.where(y_des != y_des)
        # some vector manipulation to get what we want
        breaks = breaks[1][:len(breaks[1])/2]
       
        import scipy.interpolate
        self.seqs_x = [] 
        self.seqs_y = [] 
        for ii in range(len(breaks) - 1):
            # get the ii'th sequence
            seq_x = y_des[0, breaks[ii]+1:breaks[ii+1]]
            seq_y = y_des[1, breaks[ii]+1:breaks[ii+1]]
            
            # generate function to interpolate the desired trajectory
            vals = np.linspace(0, 1, len(seq_x))
            self.seqs_x.append(scipy.interpolate.interp1d(vals, seq_x))
            self.seqs_y.append(scipy.interpolate.interp1d(vals, seq_y))

        self.y_des = [self.seqs_x[0], self.seqs_y[0]]

    def control(self, arm):
        """Drive the end-effector through a series 
           of (x,y) points"""

        if np.sum(abs(arm.x - self.target)) < .01:
            self.not_at_start = False

        if self.not_at_start or self.done:
            u = Control_OSC.control(self, arm)

        else: 
            y = np.array([self.y_des[d](self.time) for d in range(2)])
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
                    self.y_des = [self.seqs_x[self.num_seq], 
                                  self.seqs_y[self.num_seq]]
                    self.target = [self.y_des[0](0.0), self.y_des[1](0.0)]
            else: 
                self.pen_down = True
            
            self.x = arm.position(ee_only=True)
            x_des = self.gain * (y - self.x)
            u = Control_OSC.control(self, arm, x_des=x_des)

        return u
