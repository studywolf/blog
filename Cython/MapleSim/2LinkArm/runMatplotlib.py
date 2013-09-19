#Written by Travis DeWolf (Sept, 2013)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import py2LinkArm

class TwoLinkArm:
    """
    :param list u: the torque applied to each joints
    """
    def __init__(self, u = [.1, 0]): 
        self.u = np.asarray(u, dtype='float') # control signal
        self.state = np.zeros(3) # vector for current state
        self.L1=0.37 # length of arm link 1 in m
        self.L2=0.27 # length of arm link 2 in m
        self.time_elapsed = 0

        self.sim = py2LinkArm.pySim()
        self.sim.reset(self.state)
    
    def position(self):
        """Compute x,y position of the hand"""

        x = np.cumsum([0,
                       self.L1 * np.cos(self.state[1]),
                       self.L2 * np.cos(self.state[2])])
        y = np.cumsum([0,
                       self.L1 * np.sin(self.state[1]),
                       self.L2 * np.sin(self.state[2])])
        return (x, y)

    def step(self, dt):
        """Simulate the system and update the state"""
        for i in range(1000):
            self.sim.step(self.state, self.u)
        self.time_elapsed = self.state[0]

#------------------------------------------------------------
# set up initial state and global variables
arm = TwoLinkArm()
dt = 1./30 # 30 fps

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-1, 1), ylim=(-1, 1))
ax.grid()

line, = ax.plot([], [], 'o-', lw=4, mew=5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    """perform animation step"""
    global arm, dt
    arm.step(dt)
    
    line.set_data(*arm.position())
    time_text.set_text('time = %.2f' % arm.time_elapsed)
    return line, time_text

# choose the interval based on dt and the time to animate one step
from time import time
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

# frames=None for matplotlib 1.3
ani = animation.FuncAnimation(fig, animate, frames=None, #frames=np.arange(300),
                              interval=interval, blit=True, init_func=init)
plt.show()
