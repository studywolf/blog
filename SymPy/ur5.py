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
import cloudpickle
import os
import numpy as np
import sympy as sp


class robot_config:
    """ A class to calculate all the transforms and Jacobians
    for the UR5 arm. Also stores the mass of each of the links."""

    def __init__(self):

        self.num_joints = 6
        self.num_links = 7
        self.config_folder = 'ur5_config'

        # create function dictionaries
        self._T = {}  # for transform calculations
        self._J = {}  # for Jacobian calculations

        self._M = []  # placeholder for (x,y,z) inertia matrices
        self._Mq = None  # placeholder for joint space inertia matrix function
        self._Mq_g = None  # placeholder for joint space gravity term function

        # position of point of interest relative to
        # joint axes 6 (right at the origin)
        self.x = sp.Matrix([0, 0, 0, 1])
        # set up our joint angle symbols
        self.q = [sp.Symbol('q%i' % ii) for ii in range(self.num_joints)]
        self.dq = [sp.Symbol('dq%i' % ii) for ii in range(self.num_joints)]

        self.gravity = sp.Matrix([[0, 0, -9.81, 0, 0, 0]]).T

        self.joint_names = ['UR5_joint%i' % ii
                            for ii in range(self.num_joints)]

        # for the null space controller, keep arm near these angles
        self.rest_angles = np.array([0, 45, -90, 45, 90, 0])

        # create the inertia matrices for each link of the ur5
        self._M.append(np.diag([1.0, 1.0, 1.0,
                                0.02, 0.02, 0.02]))  # link0
        self._M.append(np.diag([2.5, 2.5, 2.5,
                                0.04, 0.04, 0.04]))  # link1
        self._M.append(np.diag([5.7, 5.7, 5.7,
                                0.06, 0.06, 0.04]))  # link2
        self._M.append(np.diag([3.9, 3.9, 3.9,
                                0.055, 0.055, 0.04]))  # link3
        self._M.append(np.copy(self._M[1]))  # link4
        self._M.append(np.copy(self._M[1]))  # link5
        self._M.append(np.diag([0.7, 0.7, 0.7,
                                0.01, 0.01, 0.01]))  # link6

        # segment lengths associated with each joint
        L = np.array([0.0935, 0.13453, 0.4251,
                      0.12, 0.3921, 0.0935, 0.0935, 0.0935])

        # transform matrix from origin to joint 0 reference frame
        # link 0 reference frame is the same as joint 0
        self.T0org = sp.Matrix([[sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
                                [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
                                [0, 0, 1, L[0]],
                                [0, 0, 0, 1]])

        # transform matrix from joint 0 to joint 1 reference frame
        # link 1 reference frame is the same as joint 1
        self.T10 = sp.Matrix([[1, 0, 0, -L[1]],
                              [0, sp.cos(-self.q[1] + sp.pi/2),
                               -sp.sin(-self.q[1] + sp.pi/2), 0],
                              [0, sp.sin(-self.q[1] + sp.pi/2),
                               sp.cos(-self.q[1] + sp.pi/2), 0],
                              [0, 0, 0, 1]])

        # transform matrix from joint 1 to joint 2 reference frame
        self.T21 = sp.Matrix([[1, 0, 0, 0],
                              [0, sp.cos(-self.q[2]),
                               -sp.sin(-self.q[2]), L[2]],
                              [0, sp.sin(-self.q[2]),
                               sp.cos(-self.q[2]), 0],
                              [0, 0, 0, 1]])

        # transform matrix from joint 1  to link 2
        self.Tl21 = sp.Matrix([[1, 0, 0, 0],
                               [0, sp.cos(-self.q[2]),
                                -sp.sin(-self.q[2]), L[2] / 2],
                               [0, sp.sin(-self.q[2]),
                                sp.cos(-self.q[2]), 0],
                               [0, 0, 0, 1]])

        # transform matrix from joint 2 to joint 3
        self.T32 = sp.Matrix([[1, 0, 0, L[3]],
                              [0, sp.cos(-self.q[3] - sp.pi/2),
                               -sp.sin(-self.q[3] - sp.pi/2), L[4]],
                              [0, sp.sin(-self.q[3] - sp.pi/2),
                               sp.cos(-self.q[3] - sp.pi/2), 0],
                              [0, 0, 0, 1]])

        # transform matrix from joint 2 to link 3
        self.Tl32 = sp.Matrix([[1, 0, 0, L[3]],
                               [0, sp.cos(-self.q[3] - sp.pi/2),
                                -sp.sin(-self.q[3] - sp.pi/2), L[4] / 2],
                               [0, sp.sin(-self.q[3] - sp.pi/2),
                                sp.cos(-self.q[3] - sp.pi/2), 0],
                               [0, 0, 0, 1]])

        # transform matrix from joint 3 to joint 4
        self.T43 = sp.Matrix([[sp.sin(-self.q[4] - sp.pi/2),
                               sp.cos(-self.q[4] - sp.pi/2), 0, -L[5]],
                              [sp.cos(-self.q[4] - sp.pi/2),
                               -sp.sin(-self.q[4] - sp.pi/2), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # transform matrix from joint 4 to joint 5
        self.T54 = sp.Matrix([[1, 0, 0, 0],
                              [0, sp.cos(self.q[5]), -sp.sin(self.q[5]), 0],
                              [0, sp.sin(self.q[5]), sp.cos(self.q[5]), L[6]],
                              [0, 0, 0, 1]])

        # transform matrix from joint 5 to end-effector
        self.TEE5 = sp.Matrix([[0, 0, 0, L[7]],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 1]])

        # orientation part of the Jacobian (compensating for orientations)
        self.J_orientation = [[0, 0, 10],  # joint 0 rotates around z axis
                              [10, 0, 0],  # joint 1 rotates around x axis
                              [10, 0, 0],  # joint 2 rotates around x axis
                              [10, 0, 0],  # joint 3 rotates around x axis
                              [0, 0, 10],  # joint 4 rotates around z axis
                              [1, 0, 0]]  # joint 5 rotates around x axis

    def J(self, name, q):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # check for function in dictionary
        if self._J.get(name, None) is None:
            print('Generating Jacobian function for %s' % name)
            self._J[name] = self._calc_J(name)
        return np.array(self._J[name](*q))

    def Mq(self, q):
        """ Calculates the joint space inertia matrix for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq is None:
            print('Generating inertia matrix function')
            self._Mq = self._calc_Mq()
        return np.array(self._Mq(*q))

    def Mq_g(self, q):
        """ Calculates the force of gravity in joint space for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq_g is None:
            print('Generating gravity effects function')
            self._Mq_g = self._calc_Mq_g()
        return np.array(self._Mq_g(*q)).flatten()

    def T(self, name, q):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # check for function in dictionary
        if self._T.get(name, None) is None:
            print('Generating transform function for %s' % name)
            # TODO: link0 and joint0 share a transform, but will
            # both have their own transform calculated with this check
            self._T[name] = self._calc_T(name)
        return self._T[name](*q)[:-1].flatten()

    def _calc_J(self, name, lambdify=True):
        """ Uses Sympy to generate the Jacobian for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our Jacobian saved in file
        if os.path.isfile('%s/%s.J' % (self.config_folder, name)):
            J = cloudpickle.load(open('%s/%s.J' %
                                 (self.config_folder, name), 'rb'))
        else:
            Tx = self._calc_T(name, lambdify=False)
            J = []
            # calculate derivative of (x,y,z) wrt to each joint
            for ii in range(self.num_joints):
                J.append([])
                J[ii].append(sp.simplify(Tx[0].diff(self.q[ii])))  # dx/dq[ii]
                J[ii].append(sp.simplify(Tx[1].diff(self.q[ii])))  # dy/dq[ii]
                J[ii].append(sp.simplify(Tx[2].diff(self.q[ii])))  # dz/dq[ii]

            end_point = name.strip('link').strip('joint')
            if end_point != 'EE':
                end_point = min(int(end_point) + 1, self.num_joints)
                # add on the orientation information up to the last joint
                for ii in range(end_point):
                    J[ii] = J[ii] + self.J_orientation[ii]
                # fill in the rest of the joints orientation info with 0
                for ii in range(end_point, self.num_joints):
                    J[ii] = J[ii] + [0, 0, 0]

            # save to file
            cloudpickle.dump(J, open('%s/%s.J' %
                                     (self.config_folder, name), 'wb'))

        J = sp.Matrix(J).T  # correct the orientation of J
        if lambdify is False:
            return J
        return sp.lambdify(self.q, J)

    def _calc_Mq(self, lambdify=True):
        """ Uses Sympy to generate the inertia matrix in
        joint space for the ur5

        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our inertia matrix saved in file
        if os.path.isfile('%s/Mq' % self.config_folder):
            Mq = cloudpickle.load(open('%s/Mq' % self.config_folder, 'rb'))
        else:
            # get the Jacobians for each link's COM
            J = [self._calc_J('link%s' % ii, lambdify=False)
                 for ii in range(self.num_links)]

            # transform each inertia matrix into joint space
            # sum together the effects of arm segments' inertia on each motor
            Mq = sp.zeros(self.num_joints)
            for ii in range(self.num_links):
                Mq += J[ii].T * self._M[ii] * J[ii]

            # save to file
            cloudpickle.dump(Mq, open('%s/Mq' % self.config_folder, 'wb'))

        if lambdify is False:
            return Mq
        return sp.lambdify(self.q, Mq)

    def _calc_Mq_g(self, lambdify=True):
        """ Uses Sympy to generate the force of gravity in
        joint space for the ur5

        lambdify boolean: if True returns a function to calculate
                          the Jacobian. If False returns the Sympy
                          matrix
        """

        # check to see if we have our gravity term saved in file
        if os.path.isfile('%s/Mq_g' % self.config_folder):
            Mq_g = cloudpickle.load(open('%s/Mq_g' % self.config_folder,
                                         'rb'))
        else:
            # get the Jacobians for each link's COM
            J = [self._calc_J('link%s' % ii, lambdify=False)
                 for ii in range(self.num_links)]

            # transform each inertia matrix into joint space and
            # sum together the effects of arm segments' inertia on each motor
            Mq_g = sp.zeros(self.num_joints, 1)
            for ii in range(self.num_joints):
                Mq_g += J[ii].T * self._M[ii] * self.gravity

            # save to file
            cloudpickle.dump(Mq_g, open('%s/Mq_g' % self.config_folder,
                                        'wb'))

        if lambdify is False:
            return Mq_g
        return sp.lambdify(self.q, Mq_g)

    def _calc_T(self, name, lambdify=True):  # noqa C907
        """ Uses Sympy to generate the transform for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        if os.path.isfile('%s/%s.T' % (self.config_folder, name)):
            Tx = cloudpickle.load(open('%s/%s.T' % (self.config_folder, name),
                                       'rb'))
        else:
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
            Tx = T * self.x  # to convert from transform matrix to (x,y,z)

            # save to file
            cloudpickle.dump(Tx, open('%s/%s.T' % (self.config_folder, name),
                                      'wb'))

        if lambdify is False:
            return Tx
        return sp.lambdify(self.q, Tx)
