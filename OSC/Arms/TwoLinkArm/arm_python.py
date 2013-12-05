import numpy as np

class Arm:
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, dt=1e-5, l1=.31, l2=.27): 

        self.dt = dt # timestep 
        
        # length of arm links
        self.l1=l1; self.l2=l2

        # mass of links
        m1=1.; m2=1.
        # z axis inertia moment of links
        izz1=m1/3.*self.l1**2.; izz2=m2/3.*self.l2**2
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6)); self.M2 = np.zeros((6,6)) 
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[5,5] = izz1*1000
        self.M2[0:3,0:3] = np.eye(3)*m2; self.M2[5,5] = izz2*1000

        # compute non changing constants 
        self.K1 = (1/3.*m1+m2)*self.l1**2. + 1/3.*m2*self.l2**2.; 
        self.K2 = m2*self.l1*self.l2;
        self.K3 = 1/3.*m2*self.l2**2.; 
        self.K4 = 1/2.*m2*self.l1*self.l2; 
                    
        # initial arm joint and end-effector position
        self.q = np.array([[0], [np.pi/4.]])
        self.x = self.position(ee_only=True)
        # initial arm joint and end-effector velocity
        self.dq = np.zeros(2)
        # initial arm joint and end-effector acceleration
        self.ddq = np.zeros(2)

        self.t = 0.0

    def position(self, O1=None, O2=None, O3=None, ee_only=False):
        """Compute x,y position of the hand"""
        if O1 is None: O1 = self.q[0]
        if O2 is None: O2 = self.q[1]

        x = np.cumsum([0,
                       self.l1 * np.cos(O1),
                       self.l2 * np.cos(O1+O2)])
        y = np.cumsum([0,
                       self.l1 * np.sin(O1),
                       self.l2 * np.sin(O1+O2)])
        if ee_only: return np.array([[x[-1]], [y[-1]]]).reshape(2,)
        else: return (x, y)

    def apply_torque(self, u, dt=None):
        if dt is None: 
            dt = self.dt

        # equations solved for angles
        C2 = np.cos(self.q[1])
        S2 = np.sin(self.q[1])
        M11 = (self.K1 + self.K2*C2)
        M12 = (self.K3 + self.K4*C2)
        M21 = M12
        M22 = self.K3
        H1 = -self.K2*S2*self.dq[0]*self.dq[1] - 1/2.*self.K2*S2*self.dq[1]**2.
        H2 = 1/2.*self.K2*S2*self.dq[0]**2.

        self.ddq[1] = (H2*M11 - H1*M21 - M11*u[1]+ M21*u[0]) / (M12**2. - M11*M22)
        self.ddq[0] = (-H2 + u[1]- M22*self.ddq[1]) / M21
        self.dq[1] += self.ddq[1]*dt
        self.dq[0] += self.ddq[0]*dt
        self.q[0] += self.dq[0]*dt
        self.q[1] += self.dq[1]*dt
        self.x = self.position(ee_only=True)

        # transfer to next time step 
        self.t += dt
