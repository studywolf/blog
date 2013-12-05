#Written by Travis DeWolf (November 2013)
import numpy as np

class Arm:
    """A base class for arm simulators"""

    def __init__(self, dt=1e-5, singularity_thresh=.00025):

        self.dt = dt
        self.singularity_thresh = singularity_thresh

    def apply_torque(self, u, dt):
        """Takes in a torque and timestep and updates the
        arm simulation accordingly. 

        u np.array: the control signal to apply
        dt float: the timestep
        """
        raise NotImplementedError

    def gen_jacEE(self):
        """Generates the Jacobian from end-effector to
           the origin frame"""
        raise NotImplementedError

    def gen_Mq(self):
        """Generates the mass matrix for the arm in joint space"""
        raise NotImplementedError

    def gen_Mx(self, JEE=None):
        """Generate the mass matrix in operational space"""

        Mq = self.gen_Mq()

        if JEE == None: JEE = self.gen_jacEE()
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        u,s,v = np.linalg.svd(Mx_inv)
        if len(s[abs(s) < self.singularity_thresh]) == 0: 
            # if we're not near a singularity
            Mx = np.linalg.inv(Mx_inv)
        else: 
            # in the case that the robot is near a singularity
            for i in range(len(s)):
                if s[i] < self.singularity_thresh: s[i] = 0
                else: s[i] = 1.0/float(s[i])
            Mx = np.dot(v, np.dot(np.diag(s), u.T))

        return Mx

    def position(self, q=None, ee_only=False):
        """Compute x,y position of the hand

        q list: a list of the joint angles, 
                if None use current system state
        ee_only boolean: if true only return the 
                         position of the end-effector
        """
        raise NotImplementedError
