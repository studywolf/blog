import numpy as np
import py1LinkArm

class Arm:
    def __init__(self, dt=1e-5):
        self.l1 = .8

        # mass of links
        m1=1.0
        # z axis inertia moment of links
        izz1=15.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[3:,3:] = np.eye(3)*izz1
        
        # stores information returned from maplesim
        self.state = np.zeros(3) 
        # maplesim arm simulation
        self.dt = dt
        self.sim = py1LinkArm.pySim(dt=dt)
        self.sim.reset(self.state)
        self.update_state()

    def apply_torque(self, u, dt):
        u = np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/self.dt))):
            self.sim.step(self.state, -1*u)
        self.update_state()

    def update_state(self):
        self.t = self.state[0]
        self.q = self.state[1:2]
        self.dq = self.state[2:] 

        self.x = self.position(ee_only=True)

    def position(self, q0=None, ee_only=False):
        """Compute x,y position of the hand"""
        if q0 is None: q0 = self.q[0]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)
