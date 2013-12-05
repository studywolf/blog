from control_OSC import Control_OSC
from .Trajectories import DMP as DMP
import numpy as np

class Control_DMP(Control_OSC):
    """
    A controller that uses dynamic movement primitives to 
    control a robotic arm end-effector.
    """
    def __init__(self, bfs, gain, trajectory, tau, 
                 add_to_goals=None, **kwargs): 
        """
        bfs int: the number of basis functions per DMP
        gain float: the PD gain while following a DMP trajectory
        trajectory np.array: the time series of points to follow
                             [DOFs, time], with a column of None
                             wherever the pen should be lifted
        tau float: the time scaling term
        add_to_goals np.array: floats to add to the DMP goals
                               used to scale the DMPs spatially
        """

        Control_OSC.__init__(self, **kwargs)

        self.bfs = bfs
        self.gain = gain
        self.gen_dmps(trajectory)
        self.tau = tau
        self.target,_,_ = self.dmps.step(tau=self.tau)

        if add_to_goals is not None: 
            for ii, dmp in enumerate(self.dmp_sets):
                dmp.goal[0] += add_to_goals[ii*2]
                dmp.goal[1] += add_to_goals[ii*2+1]

        self.num_seq = 0

        self.done = False
        self.not_at_start = True
      
    def gen_dmps(self, y_des):
        """Generate the DMPs necessary to follow the 
        specified trajectory.

        trajectory np.array: the time series of points to follow
                             [DOFs, time], with a column of None
                             wherever the pen should be lifted
        """

        if y_des.ndim == 1: 
            y_des = y_des.reshape(1,len(y_des))

        num_DOF = y_des.shape[0]
        # break up the trajectory into its different words
        # NaN or None signals a new word / break in drawing
        breaks = np.where(y_des != y_des)
        # some vector manipulation to get what we want
        breaks = breaks[1][:len(breaks[1])/2]
       
        self.dmp_sets = []
        for ii in range(len(breaks) - 1):
            # get the ii'th sequence
            seq = y_des[:, breaks[ii]+1:breaks[ii+1]]
            
            dmps = DMP.DMPs_discrete(dmps=num_DOF, bfs=self.bfs)
            dmps.imitate_path(y_des=seq)
            self.dmp_sets.append(dmps)
            self.target,_,_ = dmps.step(tau=.002)

        self.dmps = self.dmp_sets[0]

    def control(self, arm): 
        """Apply a given control signal in (x,y) 
           space to the arm"""

        if np.sum(abs(arm.x - self.target)) < .01:
            self.not_at_start = False

        if self.not_at_start or self.done:
            u = Control_OSC.control(self, arm)

        else:
            y,_,_ = self.dmps.step(tau=self.tau, state_fb=self.x)

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

            self.x = arm.position(ee_only=True)
            x_des = self.gain * (y - self.x)
            u = Control_OSC.control(self, arm, x_des=x_des) 

        return u
