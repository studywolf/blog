#Written by Travis DeWolf (November 2013)
import numpy as np

from CS import CanonicalSystem

class DMPs(object):
    """Implementation of Dynamic Motor Primitives, 
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(self, dmps, bfs, dt=.01,
                 y0=0, goal=1, w=None, 
                 ay=None, by=None):
        """
        dmps int: number of dynamic motor primitives
        bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """

        self.dmps = dmps 
        self.bfs = bfs 
        self.dt = dt
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.dmps)*y0 
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.dmps)*goal
        self.goal = goal 
        if w is None: 
            # default is f = 0
            w = np.zeros((self.dmps, self.bfs))
        self.w = w

        if ay is None: ay = np.ones(dmps)*25 # Schaal 2012
        self.ay = ay
        if by is None: by = self.ay.copy() / 4 # Schaal 2012
        self.by = by

        # set up the CS 
        self.cs = CanonicalSystem(dt=self.dt)
        self.timesteps = int(self.cs.run_time / self.dt)
        # set up the DMP system
        self.reset_state()

    def gen_psi_track(self): raise NotImplementedError()

    def imitate_paths(self): raise NotImplementedError() 

    def rollout(self, **kwargs):
        """Generate a system trial, no feedback is incorporated."""

        self.reset_state()
        if kwargs.has_key('tau'):
            timesteps = int(self.timesteps / kwargs['tau'])
        else: 
            timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.dmps)) 
        dy_track = np.zeros((timesteps, self.dmps))
        ddy_track = np.zeros((timesteps, self.dmps))
    
        for t in range(timesteps):
        
            y, dy, ddy = self.step(**kwargs)

            # record timestep
            y_track[t] = y
            dy_track[t] = dy
            ddy_track[t] = ddy

        return y_track, dy_track, ddy_track

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.dmps)   
        self.ddy = np.zeros(self.dmps)  
        self.cs.reset_state()

    def step(self, tau=1.0, state_fb=None):
        """Run the DMP system for a single timestep.

       tau float: scales the timestep
                  increase tau to make the system execute faster
       state_fb np.array: optional system feedback
        """

        # run canonical system
        cs_args = {'tau':tau,
                   'error_coupling':1.0}
        if state_fb is not None: 
            # take the 2 norm of the overall error
            state_fb = state_fb.reshape(1,self.dmps)
            dist = np.sqrt(np.sum((state_fb - self.y)**2))
            cs_args['error_coupling'] = 1.0 / (1.0 + 10*dist)
        x = self.cs.discrete_step(**cs_args)

        # generate basis function activation
        psi = np.exp(-self.h * (x - self.c)**2)

        for d in range(self.dmps):

            # generate the forcing term
            f = x * (self.goal[d] - self.y0[d]) * \
                (np.dot(psi, self.w[d])) / np.sum(psi)

            # DMP acceleration
            self.ddy[d] = (self.ay[d] * 
                     (self.by[d] * (self.goal[d] - self.y[d]) - \
                     self.dy[d]/tau) + f) * tau
            self.dy[d] += self.ddy[d] * tau * self.dt * cs_args['error_coupling']
            self.y[d] += self.dy[d] * self.dt * cs_args['error_coupling']

        return self.y, self.dy, self.ddy


class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs): 
        """
        """

        # call super class constructor
        super(DMPs_discrete, self).__init__(**kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.bfs) * self.bfs / self.c

        self.check_offset()
        
    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.dmps):
            if (self.y0[d] == self.goal[d]):
                self.goal[d] += 1e-4

    def gen_centers(self):
        """Set the centre of the Gaussian basis 
        functions be spaced evenly throughout run time"""

        '''x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des): 
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]'''

        # desired spacings along x
        # need to be spaced evenly between 1 and exp(-ax)
        # lowest number should be only as far as x gets 
        first = np.exp(-self.cs.ax*self.cs.run_time) 
        last = 1.05 - first
        des_c = np.linspace(first,last,self.bfs) 

        self.c = np.ones(len(des_c)) 
        for n in range(len(des_c)): 
            # x = exp(-c), solving for c
            self.c[n] = -np.log(des_c[n])
        
    def gen_psi_track(self, x_track):
        """Generates the activity of the basis functions for a given 
        canonical system rollout. 
        
        x_track array: an array storing the canonical system path
        """

        return np.exp(-self.h * (x_track[:,None] - self.c)**2)

    def imitate_path(self, y_des):
        """Takes in a desired trajectory and generates the set of 
        system parameters that best realize this path.
    
        y_des list/array: the desired trajectories of each DMP
                          should be shaped [dmps, run_time]
        """

        # set initial state and goal
        if y_des.ndim == 1: 
            y_des = y_des.reshape(1,len(y_des))
        self.y0 = y_des[:,0].copy()
        self.goal = y_des[:,-1].copy()
        self.y_des = y_des.copy()
        
        self.check_offset()

        # generate function to interpolate the desired trajectory
        import scipy.interpolate
        path = np.zeros((self.dmps, self.timesteps))
        x = np.linspace(0, self.cs.run_time, y_des.shape[1])
        for d in range(self.dmps):
            path_gen = scipy.interpolate.interp1d(x, y_des[d])
            for t in range(self.timesteps):  
                path[d, t] = path_gen(t * self.dt)
        y_des = path

        # calculate velocity of y_des
        dy_des = np.diff(y_des) / self.dt
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.dmps, 1)), dy_des))

        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / self.dt
        # add zero to the beginning of every row
        ddy_des = np.hstack((np.zeros((self.dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.dmps))
        # find the force required to move along this trajectory
        for d in range(self.dmps):
            f_target[:,d] = ddy_des[d] - self.ay[d] * \
                            (self.by[d] * (self.goal[d] - y_des[d]) - \
                            dy_des[d])
        # calculate x and psi   
        x_track = self.cs.discrete_rollout()
        psi_track = self.gen_psi_track(x_track)

        # efficiently calculate weights for BFs using weighted linear regression
        self.w = np.zeros((self.dmps, self.bfs))
        for d in range(self.dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.bfs):
                numer = np.sum(x_track * psi_track[:,b] * f_target[:,d])
                denom = np.sum(x_track**2 * psi_track[:,b])
                self.w[d,b] = numer / (k * denom)

        '''# plot the basis function activations
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(211)
        plt.plot(psi_track)
        plt.title('psi_track')

        # plot the desired forcing function vs approx
        plt.subplot(212)
        plt.plot(f_target[:,0])
        plt.plot(np.sum(psi_track * self.w[0], axis=1))
        plt.legend(['f_target', 'w*psi'])
        plt.tight_layout()
        plt.show()'''

        self.reset_state()
        return y_des


#==============================
# Test code
#==============================
if __name__ == "__main__":

    # test normal run
    dmp = DMPs_discrete(dmps=1, bfs=10, w=np.zeros((1,10)))
    y_track,dy_track,ddy_track = dmp.rollout()

    import matplotlib.pyplot as plt
    plt.figure(1, figsize=(6,3))
    plt.plot(np.ones(len(y_track))*dmp.goal, 'r--', lw=2)
    plt.plot(y_track, lw=2)
    plt.title('DMP system - no forcing term')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()

    # test imitation of path run
    plt.figure(2, figsize=(6,4))
    num_bfs = [10, 30, 50, 100, 10000]

    # a straight line to target
    path1 = np.sin(np.arange(0,1,.01)*5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[(len(path2) / 2.):] = .5 

    for ii, bfs in enumerate(num_bfs):
        dmp = DMPs_discrete(dmps=2, bfs=bfs)

        dmp.imitate_path(y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 3; dmp.goal[1] = 2

        y_track,dy_track,ddy_track = dmp.rollout()

        plt.figure(2)
        plt.subplot(211)
        plt.plot(y_track[:,0], lw=2)
        plt.subplot(212)
        plt.plot(y_track[:,1], lw=2)

    plt.subplot(211)
    a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend([a[0]], ['desired path'], loc='lower right')
    plt.subplot(212)
    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
    plt.title('DMP imitate path')
    plt.xlabel('time (ms)')
    plt.ylabel('system trajectory')
    plt.legend(['%i BFs'%i for i in num_bfs], loc='lower right')

    plt.tight_layout()
    plt.show()
