import numpy as np

class Control:
    """
    A class that holds the simulation and control dynamics for 
    a two link arm, with the dynamics carried out in Python.
    """
    def __init__(self, kp=10, kv=np.sqrt(10), # PD gain values
                       control_type='osc_x', # can be osc or gc 
                       ): 

        self.u = np.zeros((2,1)) # control signal
        self.control_type = control_type

        # gain values for PID 
        self.kp = kp; self.kv = kv
       
        # set up control function and target generation params 
        if control_type[:3] == 'osc': 
            self.control = self.control_osc
            self.target_gain = 1.6; self.target_bias = -.8
        elif control_type == 'gc': 
            self.control = self.control_gc
            self.target_gain = 2*np.pi; self.target_bias = -np.pi
        else: print 'invalid control type'; assert False

    def gen_target(self, arm):
        """Generate a target to move to"""
        self.target = np.random.random(size=(1,)) * \
            self.target_gain + self.target_bias

        if self.control_type == 'osc_x':
            return (self.target, 0)
        elif self.control_type == 'osc_y': 
            return (0, self.target)
        return arm.position(self.target)

    def set_target_from_mouse(self, target): 
        """This just takes in an (x,y) coordinate and sets 
           the target appropriately based on control style"""
        if self.control_type == 'osc_x': 
            self.target = target[0]
            return (self.target, 0)
        elif self.control_type == 'osc_y':
            self.target = target[1]
            return (0, self.target)

    def gen_jacCOM1(self, arm):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
    
        JCOM1 = np.zeros((6,1))
        JCOM1[0,0] = arm.l1 / 2. * -np.sin(arm.q[0]) 
        JCOM1[1,0] = arm.l1 / 2. * np.cos(arm.q[0]) 
        JCOM1[5,0] = 1.0

        return JCOM1

    def gen_jacEE_x(self, arm):
        """Generates the Jacobian from end-effector to
           the origin frame - for the x dimension"""

        JEE = arm.l1 * -np.sin(arm.q[0])
        
        return JEE

    def gen_jacEE_y(self, arm):
        """Generates the Jacobian from end-effector to
           the origin frame - for the y dimension"""

        JEE = arm.l1 * np.cos(arm.q[0])
        
        return JEE

    def gen_Mq(self, arm):
        """Generates the mass matrix for the arm in joint space"""
        
        # get the instantaneous Jacobians
        JCOM1 = self.gen_jacCOM1(arm)
        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(arm.M1, JCOM1))
        
        return Mq
        
    def control_gc(self, arm):
        """Generate a control signal to move the arm through
           joint space to the desired joint angle position"""
        
        # calculated desired joint angle acceleration
        prop_val = ((self.target - arm.q) + np.pi) % \
                                                    (np.pi*2) - np.pi
        q_des = (self.kp * prop_val + \
                 self.kv * -arm.dq).reshape(1,)

        Mq = self.gen_Mq(arm)

        # tau = Mq * q_des + tau_grav, but gravity = 0
        self.u = np.dot(Mq, q_des).reshape(1,)

        return self.u
        

    def control_osc(self, arm):
        """Generates a control signal to move the arm to 
           the specified target"""

        # get the instantaneous Jacobians
        if self.control_type == 'osc_x':
            JEE = self.gen_jacEE_x(arm)
            self.x = arm.position(ee_only=True)[0]
        elif self.control_type == 'osc_y':
            JEE = self.gen_jacEE_y(arm)
            self.x = arm.position(ee_only=True)[1]
        else: 
            print 'invalid control type'
            assert False

        self.dx = np.dot(JEE, arm.dq)

        # calculate desired end-effector acceleration
        x_des = (self.kp * (self.target - self.x) + \
                 self.kv * -self.dx)

        # generate the mass matrix in end-effector space
        Mq = self.gen_Mq(arm)
        Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        if abs(JEE) > .005:
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
        #import pdb; pdb.set_trace()
        Fx = Mx * x_des

        # tau = J^T * Fx + tau_grav, but gravity = 0
        self.u = np.dot(JEE.T, Fx).reshape(1,)

        return self.u
