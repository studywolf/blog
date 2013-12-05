from ..Arm import Arm
import numpy as np
import py1LinkArm

class Arm1Link(Arm):
    """A wrapper around a MapleSim generated C simulation 
    of a one link arm"""

    def __init__(self, JEE='x', **kwargs):
        """
        JEE string: specifies whether the Jacobian for OSC control of 
                    the end-effector should relay the x or y position.
        """

        Arm.__init__(self, **kwargs)


        # length of arm links
        self.l1 = .8
        self.L = np.array([self.l1])
        # mass of links
        m1=1.0
        # z axis inertia moment of links
        izz1=15.
        # create mass matrices at COM for each link
        self.M1 = np.zeros((6,6))
        self.M1[0:3,0:3] = np.eye(3)*m1; self.M1[3:,3:] = np.eye(3)*izz1

        self.resting_position = np.array([np.pi/4.0])
        
        # stores information returned from maplesim
        self.state = np.zeros(3) 
        # maplesim arm simulation
        self.sim = py1LinkArm.pySim(dt=self.dt)
        self.sim.reset(self.state)
        self.update_state()

    def apply_torque(self, u, dt):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly. 

        u np.array: the control signal to apply
        dt float: the timestep
        """
        u = np.array(u, dtype='float')
       
        for i in range(int(np.ceil(dt/self.dt))):
            self.sim.step(self.state, -1*u)
        self.update_state()

    def gen_jacCOM1(self):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
    
        JCOM1 = np.zeros((6,1))
        JCOM1[0,0] = self.l1 / 2. * -np.sin(self.q[0]) 
        JCOM1[1,0] = self.l1 / 2. * np.cos(self.q[0]) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacEE(self):
        """Generates the Jacobian from end-effector to
           the origin frame"""

        JEE = np.zeros((2,1))
        JEE[0,0] = self.l1 * -np.sin(self.q[0])
        JEE[1,0] = self.l1 * np.cos(self.q[0])
        
        return JEE

    def gen_Mq(self):
        """Generates the mass matrix for the arm in joint space"""
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1()
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(self.M1, JCOM1))
        
        return Mq
        
    def position(self, q=None, ee_only=False):
        """Compute x,y position of the hand"""
        if q is None: q0 = self.q[0]
        else: q0 = q[0]

        x = np.cumsum([0,
                       self.l1 * np.cos(q0)])
        y = np.cumsum([0,
                       self.l1 * np.sin(q0)])
        
        if ee_only: return np.array([x[-1], y[-1]])
        return (x, y)

    def update_state(self):
        """Separate out the state variable into time, angles, 
        velocities, and accelerations."""

        self.t = self.state[0]
        self.q = self.state[1:2]
        self.dq = self.state[2:] 

        self.x = self.position(ee_only=True)
