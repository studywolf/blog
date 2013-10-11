import numpy as np
import py2LinkArm

class Control:
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, kp=10, kv=np.sqrt(10), # PD gain values
                       control_type='osc', # can be osc or gc 
                       ): 

        self.u = np.zeros((2,1)) # control signal
        self.control_type = control_type

        # gain values for PID 
        self.kp = kp; self.kv = kv
       
        # set up control function and target generation params 
        if control_type == 'osc': 
            self.control = self.control_osc
            self.target_gain = .8; self.target_bias = -.4
        elif control_type == 'gc': 
            self.control = self.control_gc
            self.target_gain = 2*np.pi; self.target_bias = -np.pi
        else: print 'invalid control type'; assert False

    def gen_target(self, arm):
        self.target = np.random.random(size=(2,)) * \
            self.target_gain + self.target_bias

        if self.control_type == 'osc': return [self.target[0], self.target[1]]
        return arm.position(self.target[0], self.target[1])

    def gen_jacCOM1(self, arm):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
    
        JCOM1 = np.zeros((6,2))
        JCOM1[0,0] = arm.l1 / 2. * -np.sin(arm.q[0]) 
        JCOM1[1,0] = arm.l1 / 2. * np.cos(arm.q[0]) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self, arm):
        """Generates the Jacobian from the COM of the second 
           link to the origin frame"""

        JCOM2 = np.zeros((6,2))
        # define column entries right to left
        JCOM2[0,1] = arm.l2 / 2. * -np.sin(arm.q[0] + arm.q[1])
        JCOM2[1,1] = arm.l2 / 2. * np.cos(arm.q[0] + arm.q[1])
        JCOM2[0,0] = arm.l1 * -np.sin(arm.q[0]) + JCOM2[0,1]
        JCOM2[1,0] = arm.l1 * np.cos(arm.q[0]) + JCOM2[1,1]
        JCOM2[5,0] = 1.0; JCOM2[5,1] = 1.0

        return JCOM2

    def gen_jacEE(self, arm):
        """Generates the Jacobian from end-effector to
           the origin frame"""

        JEE = np.zeros((2,2))
        JEE[0,1] = arm.l2 * -np.sin(arm.q[0] + arm.q[1])
        JEE[1,1] = arm.l2 * np.cos(arm.q[0] + arm.q[1]) 
        JEE[0,0] = arm.l1 * -np.sin(arm.q[0]) + JEE[0,1]
        JEE[1,0] = arm.l1 * np.cos(arm.q[0]) + JEE[1,1]
        
        return JEE

    def gen_Mq(self, arm):
        """Generates the mass matrix for the arm in joint space"""
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(arm)
        JCOM2 = self.gen_jacCOM2(arm)
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(arm.M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(arm.M2, JCOM2))
        
        return Mq
        
    def control_gc(self, arm):
        """Generate a control signal to move the arm through
           joint space to the desired joint angle position"""
        
        # calculated desired joint angle acceleration
        prop_val = ((self.target.reshape(1,2) - arm.q) + np.pi) % \
                                                    (np.pi*2) - np.pi
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq + self.ka).reshape(2,)

        Mq = self.gen_Mq(arm)

        # tau = Mq * q_des + tau_grav, but gravity = 0
        self.u = np.dot(Mq, q_des).reshape(2,)

        return self.u
        

    def control_osc(self, arm):
        """Generates a control signal to move the arm to 
           the specified target"""

        # get the instantaneous Jacobians
        JEE = self.gen_jacEE(arm)

        self.x = arm.position(ee_only=True)
        self.dx = np.dot(JEE, arm.dq)[0:2]

        # calculate desired end-effector acceleration
        x_des = (self.kp * (self.target - self.x) + \
                 self.kv * -self.dx).reshape(2,1)

        # generate the mass matrix in end-effector space
        Mq = self.gen_Mq(arm)
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        if abs(np.linalg.det(np.dot(JEE,JEE.T))) > .000025:
            # if we're not near a singularity
            Mx = np.linalg.inv(Mx_inv)
        else: 
            # in the case that the robot is entering near singularity
            u,s,v = np.linalg.svd(Mx_inv)
            for i in range(len(s)):
                if s[i] < .005: s[i] = 0
                else: s[i] = 1.0/float(s[i])
            Mx = np.dot(v, np.dot(np.diag(s), u.T))

        # calculate force 
        Fx = np.dot(Mx, x_des)

        # tau = J^T * Fx + tau_grav, but gravity = 0
        self.u = np.dot(JEE.T, Fx).reshape(2,)

        return self.u
