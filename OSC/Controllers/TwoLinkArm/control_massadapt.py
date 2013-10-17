import numpy as np

class Control:

    def __init__(self, K_p=1, K_v=1, L_m=1, K_i=0, mass_adapt=False, seed_d=1):
        self.K_p = K_p
        self.K_v = K_v
        self.K_i = K_i
        self.L_m = L_m
        self.mass_adapt = mass_adapt
        self.alpha = 0.5
        self.x_d = None
        self.q_d = np.array([np.pi/2., np.pi/2.])
        self.seed_d = seed_d
        self.tau = np.array([0.,0.])
        self.Y_m = np.zeros((2,2,4))
        self.s = np.array([0.,0.])
        
        self.arm_dims = 2
        #np.array([1.105, 5.393, 1.55, 4.172]).reshape(4,1)#
        self.theta_m = np.ones((4,1), dtype='float')

    def gen_jacCOM1(self, arm):
        """Generates the Jacobian from the COM of the first
           link to the origin frame"""
    
        JCOM1 = np.zeros((3,2))
        JCOM1[0,0] = arm.l1 / 2. * -np.sin(arm.q[0]) 
        JCOM1[1,0] = arm.l1 / 2. * np.cos(arm.q[0]) 
        JCOM1[2,0] = 1.0

        return JCOM1

    def gen_jacCOM2(self, arm):
        """Generates the Jacobian from the COM of the second 
           link to the origin frame"""

        JCOM2part1 = arm.l2 / 2. * -np.sin(arm.q[0] + arm.q[1]) 
        JCOM2part2 = arm.l2 / 2. * np.cos(arm.q[0] + arm.q[1]) 

        JCOM2 = np.zeros((3,2))
        JCOM2[0,0] = arm.l1 * -np.sin(arm.q[0]) + JCOM2part1
        JCOM2[0,1] = JCOM2part1
        JCOM2[1,0] = arm.l1 * np.cos(arm.q[0]) + JCOM2part2
        JCOM2[1,1] = JCOM2part2
        JCOM2[2,0] = 1.0; JCOM2[2,1] = 1.0

        return JCOM2

    def control(self, arm, x_d_raw, q_d_raw, dt):
        if self.tau is not None and (self.tau[0]>dt or self.tau[1]>dt):
            decay = np.exp(-dt/self.tau)
            if self.x_d is None: self.x_d = x_d_raw
            if self.q_d is None: self.q_d = q_d_raw
            
            self.x_d = self.x_d*decay + x_d_raw*(1-decay)
            self.q_d = self.q_d*decay + q_d_raw*(1-decay)
        else:
            self.x_d = x_d_raw
            self.q_d = q_d_raw

        x_d = self.x_d
        q_d = self.q_d

        prop_val = (((q_d - arm.q) + np.pi) % (np.pi*2)) - np.pi
        s_old = self.s.copy()
        self.s = self.K_p*prop_val - self.K_v*arm.dq
        self.tau = self.s

        if self.mass_adapt is True: 
            # build up the Jacobians and Y_m tensor matrix
            JCOM1 = self.gen_jacCOM1(arm)
            JCOM2 = self.gen_jacCOM2(arm)

            Z1_J12 = JCOM1[:,0] * JCOM1[:,1]
            Z2_J12 = JCOM2[:,0] * JCOM2[:,1]

            Z1 = np.zeros((2,2))
            Z2 = np.zeros((2,2))
            Z3 = np.zeros((2,2))
            Z4 = np.zeros((2,2))
            # first matrix - mass of link 1
            Z1[0,0] = np.sum((JCOM1[0:2,0]**2))
            Z1[0,1] = np.sum(Z1_J12[0:2])
            Z1[1,0] = np.sum(Z1_J12[0:2])
            Z1[1,1] = np.sum((JCOM1[0:2,1]**2))
            # second matrix - Izz of link 1
            Z2[0,0] = (JCOM1[2,0]**2) 
            Z2[0,1] = Z1_J12[2]
            Z2[1,0] = Z1_J12[2]
            Z2[1,1] = (JCOM1[2,1]**2)
            # third matrix - mass of link 2
            Z3[0,0] = np.sum((JCOM2[0:2,0]**2))
            Z3[0,1] = np.sum(Z2_J12[0:2])
            Z3[1,0] = np.sum(Z2_J12[0:2])
            Z3[1,1] = np.sum((JCOM2[0:2,1]**2))
            # fourth matrix - Izz of link 2
            Z4[0,0] = (JCOM2[2,0]**2)
            Z4[0,1] = Z2_J12[2]
            Z4[1,0] = Z2_J12[2]
            Z4[1,1] = (JCOM2[2,1]**2)

            Y_m_old = self.Y_m.copy()
            self.Y_m = np.array([Z1, Z2, Z3, Z4]).T
            # our Mgc approximation
            Mgc = np.dot(self.Y_m, self.theta_m).reshape(2,2)
            tau_old = self.tau.copy()
            self.tau = np.dot(Mgc, self.s)

            dtheta_m = self.L_m*np.dot(np.dot(Y_m_old.T, tau_old), (arm.ddq)).reshape(4,1)
            self.theta_m += dtheta_m*dt
            

        return self.tau
        
        
