import numpy as np
import py3LinkArm

class Arm:
    def __init__(self, dt=1e-5):

        # length of arm links
        self.l1 = 2.0; self.l2 = 1.2; self.l3 = .7
        # mass of links
        m1=1; m2=m1; m3=m1
        # z axis inertia moment of links
        izz1=1; izz2=izz1; izz3=izz1
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M2 = np.zeros((6,6)) 
        self.M3 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[5,5] = izz1
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[5,5] = izz2
        self.M3[0:3,0:3] = np.eye(3)*m3; self.M3[5,5] = izz3
        
        # stores information returned from maplesim
        self.state = np.zeros(7) 
        # maplesim arm simulation
        self.dt = dt
        self.sim = py3LinkArm.pySim(dt=dt)
        self.sim.reset(self.state)
        self.update_state()

    def apply_torque(self, u, dt):
        u = np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/self.dt))):
            self.sim.step(self.state, -1*u)
        self.update_state()

    def update_state(self):
        self.t = self.state[0]
        self.q = self.state[1:4]
        self.dq = self.state[4:] 

        self.x = self.position(ee_only=True)

    def position(self, q0=None, q1=None, q2=None, ee_only=False):
        """Compute x,y position of the hand"""
        if q0 is None: q0 = self.q[0]
        if q1 is None: q1 = self.q[1]
        if q2 is None: q2 = self.q[2]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1), 
                       self.l3 * np.cos(q0+q1+q2)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1),
                       self.l3 * np.sin(q0+q1+q2)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)
