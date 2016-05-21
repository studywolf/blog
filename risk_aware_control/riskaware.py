import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import animation

class Runner: 

    def __init__(self):
        self.num_systems = 3
        self.num_states = 200
        self.limit = 10
        self.dx = self.limit * 2.0 / self.num_states
        self.domain = np.linspace(-self.limit, 
                                   self.limit, 
                                   self.num_states)[:,None]

        self.var = np.array([3, 1, .5])

        self.drift = 0#-.15 # systems drifts left

        # action set
        self.u = np.array([0, .1, .5, -.5])

        self.L = np.zeros((len(self.var), 
            len(self.u), self.num_states, self.num_states)) 
        for ii in range(len(self.u)):
            for jj in range(self.num_states):
                # set the system state
                self.x = np.ones(self.num_systems) * self.domain[jj]
                # get the probability distribution of x
                old_px = self.gen_px()
                # apply the control signal
                self.physics(np.ones(self.num_systems) * self.u[ii])
                # get the new probability distribution of x
                px = self.gen_px()
                # calculate the change in the probability distribution
                self.L[:, ii, :, jj] = np.copy(px - old_px).T
 
        # the initial state probability 
        self.x = np.zeros(self.num_systems)
        self.px = self.gen_px() 

        # also need a cost function (Gaussian to move towards the center)
        self.make_v()

        self.track_position = []
        self.track_target1 = []
        self.track_target2 = []

    def make_gauss(self, mean=0, var=.5):
        return np.exp(-(self.domain-mean)**2 / (2.0*var**2)) 

    def make_v(self, mean=0):
        # set up the road
        self.v = self.make_gauss(mean=mean,var=2) * 5 - 1
        self.v[np.where(self.v > .5)] = .5
        # make a preference for being in the right lane
        self.v += self.make_gauss(mean=mean+2,var=.6)

    def physics(self, u):
        self.x += (self.drift + u) # simple physics
        self.x = np.minimum(self.limit, np.maximum(-self.limit, self.x))

    def gen_px(self, x=None, var=None):
        x = np.copy(self.x) if x is None else x
        var = self.var if var is None else var

        px = self.make_gauss(x, var)
        # make sure no negative values
        px[np.where(px < 0)] = 0.0
        # make sure things sum to 1
        px /= np.sum(px, axis=0) * self.dx
        return px


    def anim_init(self):
        self.v_line.set_data([], [])
        self.px_line0.set_data([], [])
        self.px_line1.set_data([], [])
        self.px_line2.set_data([], [])
        plt.legend(['value function', 'np.dot(Li, p(x))'])
        return self.v_line, self.px_line0, self.px_line1, self.px_line2

    def anim_animate(self, i):

        # calculate the weights for the actions
        self.wu = np.zeros((self.num_systems, len(self.u)))
        self.wu2 = np.zeros((self.num_systems, len(self.u)))
      
        for ii in range(self.num_systems):
            # calculate weights for all actions simultaneously, v.T * L_i * p(x)
            # constrain so that you can only weight actions positively
            self.wu[ii] = np.maximum(self.wu[ii],
                    np.einsum('lj,ij->i', self.v.T, 
                        np.einsum('ijk,k->ij', self.L[ii], self.px[:,ii])))
            # constrain so that total output power sum_j u_j**2 = 1
            if np.sum(self.wu[ii]) != 0:
                self.wu[ii] /= np.sqrt(np.sum(self.wu[ii]**2))

        # track information for plotting
        self.track_position.append(np.copy(self.x))
        # get edges of value function
        road = np.where(self.v >= .5)
        self.track_target1.append(np.array([self.domain[road[0][0]], # left edge
                                           self.domain[road[0][-1]]])) # right edge
        lane = np.where(self.v >= .6)
        self.track_target2.append(np.array([self.domain[lane[0][0]], # left edge
                                           self.domain[lane[0][-1]]])) # right edge
        # apply the control signal, simulate dynamics and get new state
        self.physics(np.dot(self.wu, self.u))
        # get the new probability distribution of x
        self.px = self.gen_px()

        # move the target around slowly
        self.make_v(np.sin(i*.1)*5)

        self.px_line0.set_data(range(self.num_states), self.px[:,0])
        self.px_line1.set_data(range(self.num_states), self.px[:,1])
        self.px_line2.set_data(range(self.num_states), self.px[:,2])
        self.v_line.set_data(range(self.num_states), self.v)

        return self.v_line, self.px_line0, self.px_line1, self.px_line2

    def run(self): 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.v_line, = ax.plot([],[], color='r', lw=3)
        self.px_line0, = ax.plot([],[], color='k', lw=3)
        self.px_line1, = ax.plot([],[], color='k', lw=3)
        self.px_line2, = ax.plot([],[], color='k', lw=3)
        
        plt.xlim([0, self.num_states-1])
        plt.xticks(np.linspace(0, self.num_states, 11), np.linspace(-10, 10, 11))
        plt.ylim([-1, 1.5])

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                    init_func=self.anim_init, frames=500, 
                    interval=100, blit=True)

        plt.show()

if __name__ == '__main__':

    runner = Runner()
    runner.run()

    # generate some nice plots
    fig = plt.figure(figsize=(8, 8))

    # do some scaling to plot the same as seaborn heat plots
    fix_for_plot = lambda x: (np.array(x).squeeze() / 
            runner.limit / -2.0 + .5) * runner.num_states
    track_target1 = fix_for_plot(runner.track_target1)
    track_target2 = fix_for_plot(runner.track_target2)
    track_position = fix_for_plot(runner.track_position)
    runner.track_position = np.array(runner.track_position)

    time = track_position.shape[0]
    X = np.arange(0, time)
    Y = runner.domain
    X, Y = np.meshgrid(X, Y)

    plt.subplot(runner.num_systems+2, 1, 1)
    plt.title('Position on road')

    plt.subplot(runner.num_systems+2, 1, 4)
    plt.fill_between(range(track_target2.shape[0]), 
            track_target2[:,0], track_target2[:,1], facecolor='orange', alpha=.25)
    plt.fill_between(range(track_target1.shape[0]), 
            track_target1[:,0], track_target1[:,1], facecolor='y', alpha=.25)


    for ii in range(runner.num_systems):
        # plot borders, and path and heatmap for system ii
        plt.subplot(runner.num_systems+2, 1, ii+1)
        # plot road boundaries
        plt.plot(track_target1, 'r--', lw=5) 

        # plot a heat map showing sensor information
        heatmap = np.zeros((runner.num_states, time))
        for jj in range(time):
            heatmap[:,jj] = runner.make_gauss(
                    mean=runner.track_position[jj, ii], 
                    var=runner.var[ii]).flatten()
        seaborn.heatmap(heatmap, xticklabels=False, yticklabels=False, 
                cbar=False, cmap='Blues')   

        # plot filled in zones of desirability, first the road
        plt.fill_between(range(track_target1.shape[0]), 
                track_target1[:,0], track_target1[:,1], 
                facecolor='y', alpha=.25)
        # and now the lane
        plt.fill_between(range(track_target2.shape[0]), 
                track_target2[:,0], track_target2[:,1], 
                facecolor='orange', alpha=.25)

        # plot actual position of each system
        line, = plt.plot(track_position[:,ii], 'k', lw=3) 

        plt.legend([line], ['Variance = %.2f'%runner.var[ii]], 
                frameon=True, bbox_to_anchor=[1,1.05])
        plt.xlim([0, time-1])
        plt.ylabel('Position')

        # plot the borders and path of each
        ax = plt.subplot(runner.num_systems+2, 1, 4)

        plt.plot(track_position[:,ii], lw=3) # plot actual position of each system

    plt.xlim([0, time-1])
    plt.legend(runner.var, frameon=True, bbox_to_anchor=[1, 1.05])
    plt.ylabel('Position')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.subplot(runner.num_systems+2, 1, 5)
    target_center = np.sin(X[0]*.1)*5 + 2
    plt.plot((runner.track_position[:,0] - target_center)**2, lw=2)
    plt.plot((runner.track_position[:,1] - target_center)**2, lw=2)
    plt.plot((runner.track_position[:,2] - target_center)**2, lw=2)
    plt.legend(runner.var, frameon=True, bbox_to_anchor=[1, 1.05])
    plt.title('Distance from center of lane')
    plt.ylabel('Squared error')
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()
