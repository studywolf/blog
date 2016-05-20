import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import animation

class Runner: 
    """ The goal of this script is to be able to learn 
    the L operators of several actions simultaneously, 
    such that you don't have to choose only one action 
    at a time during training. """ 

    def __init__(self):
        self.num_states = 200
        self.limit = 20
        self.dx = self.limit * 2.0 / self.num_states
        self.domain = np.linspace(-self.limit, 
                                   self.limit, 
                                   self.num_states)

        self.x = 0  # initial position
        self.var = .4 # variance in sensory information
        # the initial state probability 
        self.px = self.make_gauss()

        self.drift = 1.5 # constant slide of the system in this direction 

        # action set
        self.u = np.array([0, .5, -1, 3, -5])
        # for learning, start with no knowledge about actions
        self.L = [(np.random.random((self.num_states,self.num_states))*2-1)*1e-1 \
                for ii in range(len(self.u))]
        
        # create some more for plotting later
        self.L_init = np.copy(self.L)
        self.L_actual = [np.zeros((self.num_states, self.num_states)) \
                for ii in range(len(self.u))]
        for ii in range(len(self.u)):
            for jj in range(self.num_states):
                # set the system state
                self.x = self.domain[jj]
                # get the probability distribution of x
                old_px = self.gen_px()
                # apply the control signal
                self.physics(self.u[ii])
                # get the new probability distribution of x
                px = self.gen_px()
                # calculate the change in the probability distribution
                # self.L_actual[ii][jj] = np.copy(px_old - px)
                self.L_actual[ii][:,jj] = np.copy(px - old_px)
            # self.L[ii] = np.copy(self.L_actual[ii])

        self.x = 0  # initial position
        self.gamma = 1e-1 # learning rate

        # also need a cost function (Gaussian to move towards the center)
        self.make_v()

        self.track_position = []
        self.track_target = []

    def make_gauss(self, mean=0, var=.5):
        return np.exp(-(self.domain-mean)**2 / (2*var**2)) 

    def make_v(self, mean=0):
        self.v = self.make_gauss(mean=mean,var=2) + \
                self.make_gauss(mean=mean,var=.01)
        self.v = self.v * 2 - 1
        self.v[np.where(self.v > 0)] = 1.0

    def physics(self, u):
        self.x += self.drift + u # simple physics
        self.x = min(self.limit, max(-self.limit, self.x))

    def gen_px(self, x=None, var=None, noise=False):
        x = np.copy(self.x) if x is None else x
        var = self.var if var is None else var

        if noise is True: 
            x += int(np.random.random() * 2)
            x = min(self.limit, max(-self.limit, x))
        px = self.make_gauss(x, var) 
        # make sure no negative values
        px[np.where(px < 0)] = 0.0
        # make sure things sum to 1
        px /= np.max(np.sum(px) * self.dx)
        return px

    def anim_init(self):
        self.v_line.set_data([], [])
        self.px_line.set_data([], [])
        self.Lpx_line0.set_data([], [])
        self.Lpx_line1.set_data([], [])
        self.Lpx_line2.set_data([], [])
        plt.legend(['value function', 'px', 'L0*px', 'L1*px', 'L2*px'])
        return self.v_line, self.px_line, self.Lpx_line0, self.Lpx_line1, self.Lpx_line2

    def anim_animate(self, i):

        # calculate the weights for the actions
        self.wu = np.zeros(len(self.u))
        for ii, Li in enumerate(self.L):
            # don't clip it here so we can tell the actual winner 
            self.wu[ii] = np.dot(self.v, np.dot(Li, self.px))

        # set losers to zero (there can be only one action selected)
        index = self.wu.argmax()
        # add in some exploration
        if int(np.random.random()*10) == 5: 
            index = np.random.choice(range(3))
        val = self.wu[index]
        self.wu = np.zeros(len(self.u))
        # now clip it
        self.wu[index] = 1# min(1, val)
        # store the predicted change in probability
        self.dpx_estimate = np.dot(self.L[index], self.px)
        # also store the old px for calculating dpx
        self.old_px = np.copy(self.px)

        # track information for plotting
        self.track_position.append(np.copy(self.x))
        # get edges of value function
        road = np.where(self.v == 1)
        self.track_target.append(np.array([self.domain[road[0][0]], # left edge
                                           self.domain[road[0][-1]]])) # right edge

        # simulate dynamics and get new state
        self.physics(np.dot(self.wu, self.u))
        # generate the new px
        self.px = self.gen_px(noise=False)
        # move the target around slowly
        self.make_v(np.sin(i*.01)*(self.limit-1))

        # do our learning
        err = (self.px - self.old_px) - self.dpx_estimate
        # NOTE: need to use old_px because that's what our 
        # dpx_estimate was calculated for
        learn = self.gamma * np.outer(err, self.old_px) # learning_rate * err * activities
        self.L[index] += learn

        # update the line plots
        self.px_line.set_data(range(self.num_states), self.px)
        self.Lpx_line0.set_data(range(self.num_states), np.dot(self.L[0], self.px))
        self.Lpx_line1.set_data(range(self.num_states), np.dot(self.L[1], self.px))
        self.Lpx_line2.set_data(range(self.num_states), np.dot(self.L[2], self.px))
        self.v_line.set_data(range(self.num_states), self.v)

        return self.v_line, self.px_line, self.Lpx_line0, self.Lpx_line1, self.Lpx_line2

    def run(self): 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.v_line, = ax.plot([],[], color='r', lw=3)
        self.px_line, = ax.plot([],[], color='k', lw=3)
        self.Lpx_line0, = ax.plot([],[], color='b', lw=3)
        self.Lpx_line1, = ax.plot([],[], color='g', lw=3)
        self.Lpx_line2, = ax.plot([],[], color='y', lw=3)
        
        plt.xlim([0, self.num_states-1])
        plt.xticks(np.linspace(0, self.num_states, 11), np.linspace(-10, 10, 11))
        plt.ylim([-1, 1])

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                    init_func=self.anim_init, frames=10000, 
                    interval=0, blit=True)
        plt.show()

if __name__ == '__main__':

    runner = Runner()
    runner.run()

    # generate some nice plots
    axes = []
    X, Y = np.meshgrid(runner.domain, runner.domain)
    plt.figure(figsize=(9,9))
    for ii in range(len(runner.u)):
        # axes.append(plt.subplot(1,3,ii))
        # plt.axis('equal')
        # runner.L[ii][0,0] = 1
        # runner.L[ii][0,1] = -1
        # plt.pcolormesh(X, Y, runner.L[ii])#, cmap='terrain_r')   
     
        # plot the starting vs ending vs actual L operators
        # plot a heat map showing sensor information
        axes.append(plt.subplot(4, len(runner.u), ii+1))
        plt.axis('equal')
        runner.L_init[ii][0,0] = 1
        runner.L_init[ii][0,1] = -1
        plt.pcolormesh(X, Y, runner.L_init[ii])

        axes.append(plt.subplot(4, len(runner.u), ii+1+len(runner.u)))
        runner.L[ii][0,0] = 1
        runner.L[ii][0,1] = -1
        plt.axis('equal')
        plt.pcolormesh(X, Y, runner.L[ii])

        axes.append(plt.subplot(4, len(runner.u), (ii+1+2*len(runner.u))))
        plt.axis('equal')
        runner.L_actual[ii][0,0] = 1
        runner.L_actual[ii][0,1] = -1
        plt.pcolormesh(X, Y, runner.L_actual[ii])

        axes.append(plt.subplot(4, len(runner.u), (ii+1+3*len(runner.u))))
        plt.axis('equal')
        diff = runner.L[ii] - runner.L_actual[ii]
        diff[0,0] = 1
        diff[0,1] = -1
        plt.pcolormesh(X, Y, diff)

    axes[0].set_ylabel('Initial L operator')
    axes[1].set_ylabel('Learned L operator')
    axes[2].set_ylabel('Actual L operator')
    axes[3].set_ylabel('Difference')
    plt.suptitle('Learning L operators')
    # plt.tight_layout()
    plt.show()
