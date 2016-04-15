import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import animation

class Runner: 

    def __init__(self):
        self.num_states = 400
        self.limit = 10
        self.domain = np.linspace(-self.limit, 
                                   self.limit, 
                                   self.num_states)

        self.num_systems = 3
        self.x = np.zeros(self.num_systems)
        self.var = [3, 1, .5]
        # the initial state probability 
        self.px = np.vstack([self.make_gauss() for ii in range(self.num_systems)])

        self.drift = -.15 # systems drifts left
        self.highlander_mode = False # allow more than one action at a time?

        # action set
        self.u = np.array([0, .5, -.5])

        self.L = []
        for u in self.u:
            offset = int(u / 20.0 * 400.0)
            self.L.append(
                # moves away from current state
                np.diag(np.ones(self.num_states)) * -1 +
                # moves into state + u
                np.diag(np.ones(self.num_states-abs(offset)), -offset))

        # also need a cost function (Gaussian to move towards the center)
        self.make_v()

        self.track_position = []
        self.track_target = []

    def make_gauss(self, mean=0, var=.5):
        return np.exp(-(self.domain-mean)**2 / (2*var**2)) 

    def make_v(self, mean=0):
        self.v = self.make_gauss(mean=mean,var=2) + \
                self.make_gauss(mean=mean,var=.01)
        self.v = self.v * 5 - 1
        self.v[np.where(self.v > 0)] = 1.0

    def physics(self, u):
        for ii in range(self.num_systems):
            self.x[ii] += (self.drift + u[ii]) # simple physics
            self.px[ii] = self.make_gauss(self.x[ii], self.var[ii]) 
            self.px[ii] /= np.max(self.px[ii])
            # clip at zero and normalize px
            self.px[ii][np.where(self.px[ii] < 0)] = 0.0

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
        for ii, Li in enumerate(self.L):
            for jj in range(self.num_systems):
                if self.highlander_mode is False: 
                    self.wu[jj,ii] = min(1,
                            max(0, np.dot(self.v, np.dot(Li, self.px[jj]))))
                else:
                    # don't clip it here so we can tell the actual winner 
                    self.wu[jj,ii] = np.dot(self.v, np.dot(Li, self.px[jj]))
        # select the strongest action 
        if self.highlander_mode is True:
            for ii in range(self.num_systems):
                index = self.wu[ii].argmax()
                val = self.wu[ii, index]
                self.wu[ii] = np.zeros(len(self.u))
                # now clip it
                self.wu[ii,index] = min(1, val)

        print self.wu

        # track information for plotting
        self.track_position.append(np.copy(self.x))
        # get edges of value function
        road = np.where(self.v == 1)
        self.track_target.append(np.array([self.domain[road[0][0]], # left edge
                                           self.domain[road[0][-1]]])) # right edge
        # simulate dynamics and get new state
        self.physics(np.dot(self.wu, self.u))
        # move the target around slowly
        self.make_v(np.sin(i*.1)*5)


        self.px_line0.set_data(range(self.num_states), self.px[0])
        self.px_line1.set_data(range(self.num_states), self.px[1])
        self.px_line2.set_data(range(self.num_states), self.px[2])
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
        plt.ylim([-1, 1])

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                    init_func=self.anim_init, frames=500, 
                    interval=100, blit=True)
        plt.show()

if __name__ == '__main__':

    runner = Runner()
    runner.run()

    # generate some nice plots
    fig = plt.figure(figsize=(8, 8))
    runner.track_position = np.array(runner.track_position)
    time = runner.track_position.shape[0]
    X = np.arange(0, time)
    Y = runner.domain
    X, Y = np.meshgrid(X, Y)
    for ii in range(runner.num_systems):
        # plot borders, and path and heatmap for system ii
        plt.subplot(runner.num_systems+1, 1, ii+1)
        plt.plot(runner.track_position[:,ii], 'b', lw=5) # plot actual position of each system
        plt.plot(runner.track_target, 'r--', lw=3) # plot road boundaries
        # plot a heat map showing sensor information
        heatmap = np.zeros((runner.num_states, time))
        for jj in range(time):
            heatmap[:,jj] = runner.make_gauss(mean=runner.track_position[jj, ii], 
                                              var=runner.var[ii])
        plt.pcolormesh(X, Y, heatmap, cmap='terrain_r')   


        plt.title('Variance = %.2f'%runner.var[ii])
        plt.xlim([0, time-1])

        # plot the borders and path of each
        plt.subplot(runner.num_systems+1, 1, 4)
        plt.plot(runner.track_position[:,ii], lw=3) # plot actual position of each system
    plt.plot(runner.track_target, 'r--', lw=3) # plot road boundaries
    plt.xlim([0, time-1])
    plt.legend(runner.var, frameon=True, bbox_to_anchor=[1, 1.05])
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()
