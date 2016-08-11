'''
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import sympy as sp

class calc_TnJ:
    """ A class to calculate all the transforms and Jacobians
    for the UR5 arm. Also stores the mass of each of the links."""

    def __init__(self):

        self.num_joints = 6

        # create function dictionaries
        self._T = {} # for transform calculations
        self._J = {} # for Jacobian calculations

    def calc_T(self, name, lambdify=True):
        """ Uses Sympy to generate the transform for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate 
                          the transform. If False returns the Sympy
                          matrix 
        """

        if name == 'joint0' or name == 'link0':
            T = self.T0org
        elif name == 'joint1' or name == 'link1':
            T = self.T0org * self.T10
        elif name == 'joint2':
            T = self.T0org * self.T10 * self.T21
        elif name == 'link2': 
            T = self.T0org * self.T10 * self.Tl21
        elif name == 'joint3':
            T = self.T0org * self.T10 * self.T21 * self.T32
        elif name == 'link3':
            T = self.T0org * self.T10 * self.T21 * self.Tl32
        elif name == 'joint4' or name == 'link4':
            T = self.T0org * self.T10 * self.T21 * self.T32 * self.T43
        elif name == 'joint5' or name == 'link5':
            T = self.T0org * self.T10 * self.T21 * self.T32 * self.T43 * \
                    self.T54
        elif name == 'link6' or name == 'EE':
            T = self.T0org * self.T10 * self.T21 * self.T32 * self.T43 * \
                    self.T54 * self.TEE5
        Tx = T * self.x # to convert from transform matrix to (x,y,z) 

        if lambdify is False:
            return Tx
        return sp.lambdify(self.q, Tx)

    def calc_J(self, name, lambdify=True):
        """ Uses Sympy to generate the Jacobian for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate 
                          the Jacobian. If False returns the Sympy
                          matrix 
        """

        Tx = self.calc_T(name, lambdify=False)
        J = []
        # calculate derivative of (x,y,z) wrt to each joint
        for ii in range(self.num_joints):
            J.append([])
            J[ii].append(sp.simplify(Tx[0].diff(self.q[ii])))
            J[ii].append(sp.simplify(Tx[1].diff(self.q[ii])))
            J[ii].append(sp.simplify(Tx[2].diff(self.q[ii])))
        
        end_point = name.strip('link').strip('joint')
        if end_point != 'EE':
            end_point = min(int(end_point) + 1, self.num_joints)
            # add on the orientation information up to the last joint
            for ii in range(end_point):
                J[ii] = J[ii] + self.J_orientation[ii]
            # fill in the rest of the joints orientation info with 0
            for ii in range(end_point, self.num_joints):
                J[ii] = J[ii] + [0, 0, 0]

        if lambdify is False: 
            return J
        return sp.lambdify(self.q, J)

    def T(self, name, q):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles 
        """
        # check for function in dictionary 
        if self._T.get(name, None) is None:
            print('Generating transform function for %s'%name)
            self._T[name] = self.calc_T(name)
        return self._T[name](*q)[:-1].flatten()

    def J(self, name, q):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles 
        """
        # check for function in dictionary 
        if self._J.get(name, None) is None:
            print('Generating Jacobian function for %s'%name)
            self._J[name] = self.calc_J(name)
        return np.array(self._J[name](*q)).T
 
    # position of point of interest relative to joint axes 6 (right at the origin)
    x = sp.Matrix([0,0,0,1]) 
    # set up our joint angle symbols (6th angle doesn't affect any kinematics)
    q = [sp.Symbol('q%i'%ii) for ii in range(6)]

    # segment lengths associated with each joint 
    L = np.array([0.0935, .13453, .4251, .12, .3921, .0935, .0935, .0935]) 

    # create the inertia matrices for each link of the ur5
    M_link0 = np.diag([1.0, 1.0, 1.0, 0.02, 0.02, 0.02])
    M_link1 = np.diag([2.5, 2.5, 2.5, 0.04, 0.04, 0.04])
    M_link2 = np.diag([5.7, 5.7, 5.7, 0.06, 0.06, 0.04])
    M_link3 = np.diag([3.9, 3.9, 3.9, .055, .055, 0.04])
    M_link4 = np.copy(M_link1)
    M_link5 = np.copy(M_link1)
    M_link6 = np.diag([0.7, 0.7, 0.7, 0.01, 0.01, 0.01])

    # calculate the force of gravity for each link of the ur5
    gravity = np.array([0, 0, -9.81, 0, 0, 0])
    M_link0_g = np.dot(M_link0, gravity)
    M_link1_g = np.dot(M_link1, gravity)
    M_link2_g = np.dot(M_link2, gravity)
    M_link3_g = np.dot(M_link3, gravity)
    M_link4_g = np.dot(M_link4, gravity)
    M_link5_g = np.dot(M_link5, gravity)
    M_link6_g = np.dot(M_link6, gravity)

    # transform matrix from origin to joint 0 reference frame
    # link 0 reference frame is the same as joint 0
    T0org = sp.Matrix([
        [sp.cos(q[0]), -sp.sin(q[0]), 0, 0,],
        [sp.sin(q[0]), sp.cos(q[0]), 0, 0],
        [0, 0, 1, L[0]],
        [0, 0, 0, 1]])
    
    # transform matrix from joint 0 to joint 1 reference frame
    # link 1 reference frame is the same as joint 1
    T10 = sp.Matrix([
        [1, 0, 0, -L[1]],
        [0, sp.cos(-q[1] + sp.pi/2), -sp.sin(-q[1] + sp.pi/2), 0], 
        [0, sp.sin(-q[1] + sp.pi/2), sp.cos(-q[1] + sp.pi/2), 0], 
        [0, 0, 0, 1]])

    # transform matrix from joint 1 to joint 2 reference frame
    T21 = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(-q[2]), -sp.sin(-q[2]), L[2]],
        [0, sp.sin(-q[2]), sp.cos(-q[2]), 0],
        [0, 0, 0, 1]]) 

    # transform matrix from joint 1  to link 2
    Tl21 = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(-q[2]), -sp.sin(-q[2]), L[2] / 2],
        [0, sp.sin(-q[2]), sp.cos(-q[2]), 0],
        [0, 0, 0, 1]]) 

    # transform matrix from joint 2 to joint 3
    T32 = sp.Matrix([
        [1, 0, 0, L[3]],
        [0, sp.cos(-q[3] - sp.pi/2), -sp.sin(-q[3] - sp.pi/2), L[4]],
        [0, sp.sin(-q[3] - sp.pi/2), sp.cos(-q[3] - sp.pi/2), 0],
        [0, 0, 0, 1]])

    # transform matrix from joint 2 to link 3
    Tl32 = sp.Matrix([
        [1, 0, 0, L[3]],
        [0, sp.cos(-q[3] - sp.pi/2), -sp.sin(-q[3] - sp.pi/2), L[4] / 2],
        [0, sp.sin(-q[3] - sp.pi/2), sp.cos(-q[3] - sp.pi/2), 0],
        [0, 0, 0, 1]])

    # transform matrix from joint 3 to joint 4
    T43 = sp.Matrix([
        [sp.sin(-q[4] - sp.pi/2), sp.cos(-q[4] - sp.pi/2), 0, -L[5]],
        [sp.cos(-q[4] - sp.pi/2), -sp.sin(-q[4] - sp.pi/2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    # transform matrix from joint 4 to joint 5
    T54 = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(q[5]), -sp.sin(q[5]), 0],
        [0, sp.sin(q[5]), sp.cos(q[5]), L[6]],
        [0, 0, 0, 1]])

    # transform matrix from joint 5 to end-effector
    TEE5 = sp.Matrix([
        [0, 0, 0, L[7]],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]])

    # orientation part of the Jacobian (compensating for orientations)
    J_orientation = [
        [0, 0, 1], # joint 0 rotates around z axis
        [1, 0, 0], # joint 1 rotates around x axis
        [1, 0, 0], # joint 2 rotates around x axis
        [1, 0, 0], # joint 3 rotates around x axis
        [0, 0, 1], # joint 4 rotates around z axis 
        [1, 0, 0]] # joint 5 rotates around x axis

if __name__ == '__main__':

    import numpy as np
    cTJ = calc_TnJ()
    print(cTJ.calc_J('EE', lambdify=True)(np.zeros(cTJ.num_joints)))
