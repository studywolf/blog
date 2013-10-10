import numpy as np
import py2LinkArm

class Arm:
    def __init__(self, dt=1e-5):
        self.l1 = .31
        self.l2 = .27

        # mass of links
        m1=1.98; m2=1.32
        # z axis inertia moment of links
        izz1=15.; izz2=8.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6)); self.M2 = np.zeros((6,6)) 
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[3:,3:] = np.eye(3)*izz1
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[3:,3:] = np.eye(3)*izz2
        
        # stores information returned from maplesim
        self.state = np.zeros(7) 
        # maplesim arm simulation
        self.dt = dt
        self.sim = py2LinkArm.pySim(dt=dt)
        self.sim.reset(self.state)
        self.update_state()

    def apply_torque(self, u, dt):
        u = np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/self.dt))):
            self.sim.step(self.state, u)
        self.update_state()

    def is_near_singularity(self, threshold=.1):
        """Checks the current configuration of the arm
        to see if we're within threshold distance of a singularity"""
        
        if abs(self.q[1]) <= threshold or \
           abs(self.q[1] - np.pi) <= threshold: return True
        return False
   
    def update_state(self):
        self.t = self.state[0]
        self.q = self.state[1:3]
        self.dq = self.state[3:5] 
        self.ddq = self.state[5:] 

        self.x = self.position(ee_only=True)

    def position(self, q0=None, q1=None, ee_only=False):
        """Compute x,y position of the hand"""
        if q0 is None: q0 = self.q[0]
        if q1 is None: q1 = self.q[1]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0),
                       self.l2 * np.cos(q0+q1)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0),
                       self.l2 * np.sin(q0+q1)])
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)
