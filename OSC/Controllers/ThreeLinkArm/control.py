import numpy as np

class Control:
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, kp=10, kv=np.sqrt(10), # PD gain values
                       control_type='osc', # can be osc or gc 
                       thresh = .00025): # singularity threshold

        self.u = np.zeros((2,1)) # control signal
        self.control_type = control_type
        self.thresh = thresh

        # gain values for PID 
        self.kp = kp; self.kv = kv
       
        # set up control function and target generation params 
        if control_type == 'osc': 
            self.control = self.control_osc
            self.target_gain = 6.; self.target_bias = -3.
        elif control_type == 'osc_and_null':
            self.control = self.control_osc_and_null
            self.target_gain = 6.; self.target_bias = -3.
        elif control_type == 'gc': 
            self.control = self.control_gc
            self.target_gain = 2*np.pi; self.target_bias = -np.pi
        else: print 'invalid control type'; assert False

        np.random.seed(1)

    def gen_target(self, arm):
        """Generate a target based on the control type"""

        if self.control_type == 'osc' or self.control_type == 'osc_and_null':
            self.target = np.random.random(size=(2,)) * \
                self.target_gain + self.target_bias
            return [self.target[0], self.target[1]]
        # else we need three target angles
        self.target = np.random.random(size=(3,)) * \
            self.target_gain + self.target_bias
        return arm.position(self.target[0], self.target[1], self.target[2])
        
    def set_target_from_mouse(self, target):
        """Takes in a (x,y) coordinate and sets the target"""
        self.target = target
        return self.target

    def gen_jacCOM1(self, arm):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
        q0 = arm.q[0]
    
        JCOM1 = np.zeros((6,3))
        JCOM1[0,0] = arm.l1 / 2. * -np.sin(q0) 
        JCOM1[1,0] = arm.l1 / 2. * np.cos(q0) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self, arm):
        """Generates the Jacobian from the COM of the second 
           link to the origin frame"""
        q0 = arm.q[0]
        q01 = arm.q[0] + arm.q[1]

        JCOM2 = np.zeros((6,3))
        # define column entries right to left
        JCOM2[0,1] = arm.l2 / 2. * -np.sin(q01)
        JCOM2[1,1] = arm.l2 / 2. * np.cos(q01)
        JCOM2[5,1] = 1.0

        JCOM2[0,0] = arm.l1 * -np.sin(q0) + JCOM2[0,1]
        JCOM2[1,0] = arm.l1 * np.cos(q0) + JCOM2[1,1]
        JCOM2[5,0] = 1.0

        return JCOM2
    
    def gen_jacCOM3(self, arm): 
        """Generates the Jacobian from the COM of the third
           link to the origin frame"""
        q0 = arm.q[0]
        q01 = arm.q[0] + arm.q[1] 
        q012 = arm.q[0] + arm.q[1] + arm.q[2]

        JCOM3 = np.zeros((6,3))
        # define column entries right to left
        JCOM3[0,2] = arm.l3 / 2. * -np.sin(q012)
        JCOM3[1,2] = arm.l3 / 2. * np.cos(q012)
        JCOM3[5,2] = 1.0

        JCOM3[0,1] = arm.l2 * -np.sin(q01) + JCOM3[0,2]
        JCOM3[1,1] = arm.l2 * np.cos(q01) + JCOM3[1,2]
        JCOM3[5,1] = 1.0 

        JCOM3[0,0] = arm.l1 * -np.sin(q0) + JCOM3[0,1]
        JCOM3[1,0] = arm.l1 * np.cos(q0) + JCOM3[1,1]
        JCOM3[5,0] = 1.0 

        return JCOM3

    def gen_jacEE(self, arm):
        """Generates the Jacobian from end-effector to
           the origin frame"""
        q0 = arm.q[0]
        q01 = arm.q[0] + arm.q[1] 
        q012 = arm.q[0] + arm.q[1] + arm.q[2]

        JEE = np.zeros((2,3))
        # define column entries right to left
        JEE[0,2] = arm.l3 * -np.sin(q012)
        JEE[1,2] = arm.l3 * np.cos(q012)

        JEE[0,1] = arm.l2 * -np.sin(q01) + JEE[0,2]
        JEE[1,1] = arm.l2 * np.cos(q01) + JEE[1,2]

        JEE[0,0] = arm.l1 * -np.sin(q0) + JEE[0,1]
        JEE[1,0] = arm.l1 * np.cos(q0) + JEE[1,1]

        return JEE

    def gen_djacEE(self, arm):
        """Generates the Jacobian from end-effector to
           the origin frame"""
        q0 = arm.q[0]
        dq0 = arm.dq[0]
        q01 = arm.q[0] + arm.q[1] 
        dq01 = arm.dq[0] + arm.dq[1]
        q012 = arm.q[0] + arm.q[1] + arm.q[2]
        dq012 = arm.dq[0] + arm.dq[1] + arm.dq[2]

        dJEE = np.zeros((2,3))
        # define column entries right to left
        dJEE[0,2] = dq012 * arm.l3 * -np.cos(q012)
        dJEE[1,2] = dq012 * arm.l3 * -np.sin(q012)

        dJEE[0,1] = dq01 * arm.l2 * -np.cos(q01) + dJEE[0,2]
        dJEE[1,1] = dq01 * arm.l2 * -np.sin(q01) + dJEE[1,2]

        dJEE[0,0] = dq0 * arm.l1 * -np.cos(q0) + dJEE[0,1]
        dJEE[1,0] = dq0 * arm.l1 * -np.sin(q0) + dJEE[1,1]

        return dJEE

    def gen_Mq(self, arm):
        """Generates the mass matrix of the arm in joint space"""

        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(arm)
        JCOM2 = self.gen_jacCOM2(arm)
        JCOM3 = self.gen_jacCOM3(arm)
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(arm.M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(arm.M2, JCOM2)) + \
             np.dot(JCOM3.T, np.dot(arm.M3, JCOM3))

        return Mq

    def gen_Mx(self, arm, thresh=None):
        """Generate the mass matrix in operational space"""
        if thresh is None: thresh = self.thresh

        Mq = self.gen_Mq(arm)

        JEE = self.gen_jacEE(arm)
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        u,s,v = np.linalg.svd(Mx_inv)
        if s[abs(s)<thresh].shape[0] == 0: 
            # if we're not near a singularity
            Mx = np.linalg.inv(Mx_inv)
        else: 
            # in the case that the robot is near a singularity
            for i in range(len(s)):
                if s[i] < thresh: s[i] = 0
                else: s[i] = 1.0/float(s[i])
            Mx = np.dot(v, np.dot(np.diag(s), u.T))

        return Mx, Mx_inv
        
    def control_gc(self, arm):
        """Generate a control signal to move the arm through
           joint space to the desired joint angle position"""
        
        # calculated desired joint angle acceleration
        prop_val = ((self.target.reshape(1,3) - arm.q) + np.pi) % \
                                                    (np.pi*2) - np.pi
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq).reshape(3,)

        Mq = self.gen_Mq(arm)

        # tau = Mq * q_des + tau_grav, but gravity = 0
        self.u = np.dot(Mq, q_des).reshape(3,)

        return self.u
        

    def control_osc(self, arm):
        """Generates a control signal to move the arm to 
           the specified target"""

        # get the instantaneous Jacobian for the end-effector
        JEE = self.gen_jacEE(arm)

        self.x = arm.position(ee_only=True)
        #self.dx = np.dot(JEE, arm.dq)[0:2]

        # calculate desired end-effector acceleration
        x_des = (self.kp * (self.target - self.x))# + \
                 #self.kv * -self.dx).reshape(2,1)

        # generate the mass matrix in end-effector space
        Mq = self.gen_Mq(arm)
        Mx,_ = self.gen_Mx(arm)

        # calculate force 
        Fx = np.dot(Mx, x_des)

        # tau = J^T * Fx + tau_grav, but gravity = 0
        self.u = np.dot(JEE.T, Fx).reshape(3,) - np.dot(Mq, self.kv * arm.dq)

        return self.u

    def control_osc_and_null(self, arm): 
        """Generates the control signal that moves the end-effector
           to a target location, and in the null space keeps the arm 
           joints near their resting configuration"""
    
        # get our primary control signal
        u = self.control_osc(arm)

        # calculate our secondary control signal
        # calculated desired joint angle acceleration
        rest_angles = np.array([np.pi/4., np.pi/4, np.pi/4])
        prop_val = ((rest_angles - arm.q) + np.pi) % (np.pi*2) - np.pi
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq).reshape(3,)

        Mq = self.gen_Mq(arm)
        u_null = np.dot(Mq, q_des)

        # calculate the null space filter
        JEE = self.gen_jacEE(arm)
        Mx, Mx_inv = self.gen_Mx(arm)
        Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
        null_filter = np.eye(3) - np.dot(JEE.T, Jdyn_inv)

        null_signal = np.dot(null_filter, u_null).reshape(3,)

        self.u = u + null_signal

        return self.u

