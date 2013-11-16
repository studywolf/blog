#Written by Travis DeWolf (November 2013)
import numpy as np

from CS import CanonicalSystem

class DMPs(object):
    """Implementation of Dynamic Motor Primitives, 
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(self, dmps, bfs, dt=.01,
                 y0=0, goal=1, w=None, 
                 ay=None, by=None, 
                 **kwargs): # CS parameters
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
        self.cs = CanonicalSystem(**kwargs)

    def gen_psi_track(self): raise NotImplementedError()

    def imitate_paths(self): raise NotImplementedError() 

    def open_rollout(self):
        """Generate a system trial. Canonical system is run offline
        and no feedback is incorporated.

        """
        timesteps = int(self.run_time / self.dt)

        # run canonical system, record activations
        x_track = self.cs.discrete_open(dt=self.dt, run_time=self.run_time)
        psi_track = self.gen_psi_track(x_track=x_track)

        # set system state
        y = self.y0.copy()
        dy = np.zeros(self.dmps)   
        ddy = np.zeros(self.dmps)   

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.dmps)) # desired path
        dy_track = np.zeros((timesteps, self.dmps)) # desired velocity
        ddy_track = np.zeros((timesteps, self.dmps)) # desired acceleration
    
        for t in range(timesteps):
        
            for d in range(self.dmps):
                # generate the forcing term
                f = x_track[t] * (self.goal[d] - self.y0[d]) * \
                        (np.dot(psi_track[t], self.w[d])) / np.sum(psi_track[t])

                # DMP acceleration
                ddy[d] = self.tau * (self.ay[d] * 
                         (self.by[d] * (self.goal[d] - y[d]) - dy[d]) + f)
                dy[d] += self.tau * ddy[d] * self.dt
                y[d] += self.tau * dy[d] * self.dt 

            # record timestep
            y_track[t] = y
            dy_track[t] = dy
            ddy_track[t] = ddy

        return y_track, dy_track, ddy_track

class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, run_time=5, **kwargs): 
        """
        run_time float: change how long the movement takes
        """

        self.run_time = run_time
        # tau should be set to 1 / total_time
        # to ensure movement is carried out in desired length of time
        self.tau = 1.0 / run_time

        # call super class constructor
        super(DMPs_discrete, self).__init__(**kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        #h = np.ones(self.bfs) * self.bfs * 1.0/c**2
        self.h = np.ones(self.bfs) * self.bfs**2 / self.c**2

        self.check_offset()
        
    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.dmps):
            if (self.y0[d] == self.goal[d]):
                self.goal[d] += 1e-4

    def gen_centers(self):
        """Set the centre of the Gaussian basis 
        functions be spaced evenly between 1 and 0"""

        x_track = self.cs.discrete_open(dt=self.dt, run_time=self.run_time)
        t = np.arange(len(x_track))*self.dt
        c_des = np.arange(0, 1, 1./self.bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des): 
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]

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
        self.y0 = y_des[:,0].copy()
        self.goal = y_des[:,-1].copy()
        self.y_des = y_des.copy()
        
        self.run_time = y_des.shape[1]*self.dt
        self.tau = 1.0 / self.run_time
        self.gen_centers() # generate centers for new run_time

        self.check_offset()

        # calculate velocity of y_des
        dy_des = np.diff(y_des) / float(self.dt)
        # add zero to end of every row
        #dy_des = np.hstack((dy_des, np.zeros((self.dmps, 1))))
        dy_des = np.hstack((np.zeros((self.dmps, 1)), dy_des))

        # calculate acceleration of y_des
        ddy_des = np.diff(dy_des) / float(self.dt)
        # add zero to end of every row
        #ddy_des = np.hstack((ddy_des, np.zeros((self.dmps, 1))))
        ddy_des = np.hstack((np.zeros((self.dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.dmps))
        # find the force required to move along this trajectory
        for d in range(self.dmps):
            f_target[:,d] = ddy_des[d] / self.tau**2 - self.ay[d] * \
                            (self.by[d] * (self.goal[d] - y_des[d]) - \
                            dy_des[d] / self.tau)
        # calculate x and psi
        x_track = self.cs.discrete_open(dt=self.dt, run_time=self.run_time)
        psi_track = self.gen_psi_track(x_track)

        self.w = np.zeros((self.dmps, self.bfs))
        for d in range(self.dmps):
            for b in range(self.bfs):
                # diminishing and spatial scaling term
                s = x_track * (self.goal[d] - self.y0[d])
                # BF activation through time
                G = np.diag(psi_track[:,b])
                # weighted BF activation
                sG = np.dot(s, G)
                print np.dot(sG, s)
                # weighted linear regression solution
                self.w[d,b] = np.dot(sG, f_target[:,d]) / \
                                (np.dot(sG, s) + 1e-10)

        '''
        # plot the basis function activations
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(211)
        plt.plot(psi_track)

        # plot the desired forcing function vs approx
        plt.subplot(212)
        plt.plot(f_target)
        plt.plot(np.sum(psi_track * self.w, axis=1))
        plt.show()
        '''


#==============================
# Test code
#==============================
if __name__ == "__main__":

    # test normal run
    dmp = DMPs_discrete(dmps=1, bfs=10, w=np.zeros((1,10)))
    y_track,dy_track,ddy_track = dmp.open_rollout()

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
    num_bfs = [10, 30, 50, 100]

    # a straight line to target
    path1 = np.sin(np.arange(0,1,.01)*5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[(len(path2) / 2.):] = .5 

    for ii, bfs in enumerate(num_bfs):
        dmp = DMPs_discrete(dmps=2, bfs=bfs, w=np.random.random((1,10)))

        dmp.imitate_path(y_des=np.array([path1, path2]))
        # change the scale of the movement
        dmp.goal[0] = 3; dmp.goal[1] = 2

        y_track,dy_track,ddy_track = dmp.open_rollout()

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
