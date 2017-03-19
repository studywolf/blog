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
        self._Tx = {}  # for transform calculations
        self._T_inv = {}  # for inverse transform calculations
        self._J = {}  # for Jacobian calculations

        self._M = []  # placeholder for (x,y,z) inertia matrices
        self._Mq = None  # placeholder for joint space inertia matrix function
        self._Mq_g = None  # placeholder for joint space gravity term function

        # set up our joint angle symbols
        self.q = [sp.Symbol('q%i' % ii) for ii in range(self.num_joints)]
        self.dq = [sp.Symbol('dq%i' % ii) for ii in range(self.num_joints)]
        # set up an (x,y,z) offset
        self.x = [sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')]

        self.gravity = sp.Matrix([[0, 0, -9.81, 0, 0, 0]]).T

        self.joint_names = ['UR5_joint%i' % ii
                            for ii in range(self.num_joints)]

        # for the null space controller, keep arm near these angles
        self.rest_angles = np.array([None,
                                     np.pi/4.0,
                                     -np.pi/2.0,
                                     np.pi/4.0,
                                     np.pi/2.0,
                                     np.pi/2.0])

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
        self.Torg0 = sp.Matrix([
            [sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
            [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
            [0, 0, 1, L[0]],
            [0, 0, 0, 1]])

        # transform matrix from joint 0 to joint 1 reference frame
        # link 1 reference frame is the same as joint 1
        self.T01 = sp.Matrix([
            [1, 0, 0, -L[1]],
            [0, sp.cos(-self.q[1] + sp.pi/2),
             -sp.sin(-self.q[1] + sp.pi/2), 0],
            [0, sp.sin(-self.q[1] + sp.pi/2),
             sp.cos(-self.q[1] + sp.pi/2), 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 1 to joint 2 reference frame
        self.T12 = sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(-self.q[2]),
             -sp.sin(-self.q[2]), L[2]],
            [0, sp.sin(-self.q[2]),
             sp.cos(-self.q[2]), 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 1  to link 2
        self.T1l2 = sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(-self.q[2]),
             -sp.sin(-self.q[2]), L[2] / 2],
            [0, sp.sin(-self.q[2]),
             sp.cos(-self.q[2]), 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 2 to joint 3
        self.T23 = sp.Matrix([
            [1, 0, 0, L[3]],
            [0, sp.cos(-self.q[3] - sp.pi/2),
             -sp.sin(-self.q[3] - sp.pi/2), L[4]],
            [0, sp.sin(-self.q[3] - sp.pi/2),
             sp.cos(-self.q[3] - sp.pi/2), 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 2 to link 3
        self.T2l3 = sp.Matrix([
            [1, 0, 0, L[3]],
            [0, sp.cos(-self.q[3] - sp.pi/2),
             -sp.sin(-self.q[3] - sp.pi/2), L[4] / 2],
            [0, sp.sin(-self.q[3] - sp.pi/2),
             sp.cos(-self.q[3] - sp.pi/2), 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 3 to joint 4
        self.T34 = sp.Matrix([
            [sp.sin(-self.q[4] - sp.pi/2),
             sp.cos(-self.q[4] - sp.pi/2), 0, -L[5]],
            [sp.cos(-self.q[4] - sp.pi/2),
             -sp.sin(-self.q[4] - sp.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        # transform matrix from joint 4 to joint 5
        self.T45 = sp.Matrix([
            [1, 0, 0, 0],
            [0, sp.cos(-self.q[5]), -sp.sin(-self.q[5]), 0],
            [0, sp.sin(-self.q[5]), sp.cos(-self.q[5]), L[6]],
            [0, 0, 0, 1]])

        # transform matrix from joint 5 to end-effector
        self.T5EE = sp.Matrix([
            [1, 0, 0, L[7]],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        # orientation part of the Jacobian (compensating for orientations)
        self.J_orientation = [[0, 0, 10],  # joint 0 rotates around z axis
                              [10, 0, 0],  # joint 1 rotates around x axis
                              [10, 0, 0],  # joint 2 rotates around x axis
                              [10, 0, 0],  # joint 3 rotates around x axis
                              [0, 0, 10],  # joint 4 rotates around z axis
                              [1, 0, 0]]  # joint 5 rotates around x axis

    def J(self, name, q, x=[0, 0, 0]):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q np.array: joint angles
        """
        # check for function in dictionary
        if self._J.get(name, None) is None:
            print('Generating Jacobian function for %s' % name)
            self._J[name] = self._calc_J(
                name, x=x)
        parameters = tuple(q) + tuple(x)
        return np.array(self._J[name](*parameters))

    def Mq(self, q):
        """ Calculates the joint space inertia matrix for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq is None:
            print('Generating inertia matrix function')
            self._Mq = self._calc_Mq()
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._Mq(*parameters))

    def Mq_g(self, q):
        """ Calculates the force of gravity in joint space for the ur5

        q np.array: joint angles
        """
        # check for function in dictionary
        if self._Mq_g is None:
            print('Generating gravity effects function')
            self._Mq_g = self._calc_Mq_g()
        parameters = tuple(q) + (0, 0, 0)
        return np.array(self._Mq_g(*parameters)).flatten()

    def Tx(self, name, q, x=[0, 0, 0]):
        """ Calculates the transform for a joint or link

        name string: name of the joint or link, or end-effector
        q list: set of joint angles to pass in to the T function
        x list: the [x,y,z] position of interest in "name"'s reference frame
        """
        # check for function in dictionary
        if self._Tx.get(name, None) is None:
            print('Generating transform function for %s' % name)
            # TODO: link0 and joint0 share a transform, but will
            # both have their own transform calculated with this check
            self._Tx[name] = self._calc_Tx(
                name, x=x)
        parameters = tuple(q) + tuple(x)
        return self._Tx[name](*parameters)[:-1].flatten()

    def T_inv(self, name, q, x=[0, 0, 0]):
        """ Calculates the inverse transform for a joint or link

        q list: set of joint angles to pass in to the T function
        """
        # check for function in dictionary
        if self._T_inv.get(name, None) is None:
            print('Generating inverse transform function for % s' % name)
            self._T_inv[name] = self._calc_T_inv(
                name=name, x=x)
        parameters = tuple(q) + tuple(x)
        return self._T_inv[name](*parameters)


    def _calc_J(self, name, x, lambdify=True):
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
            Tx = self._calc_Tx(name, x=x, lambdify=False)
            J = []
            # calculate derivative of (x,y,z) wrt to each joint
            for ii in range(self.num_joints):
                J.append([])
                J[ii].append(Tx[0].diff(self.q[ii]))  # dx/dq[ii]
                J[ii].append(Tx[1].diff(self.q[ii]))  # dy/dq[ii]
                J[ii].append(Tx[2].diff(self.q[ii]))  # dz/dq[ii]

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
        return sp.lambdify(self.q + self.x, J)

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
            J = [self._calc_J('link%s' % ii, x=[0, 0, 0], lambdify=False)
                 for ii in range(self.num_links)]

            # transform each inertia matrix into joint space
            # sum together the effects of arm segments' inertia on each motor
            Mq = sp.zeros(self.num_joints)
            for ii in range(self.num_links):
                Mq += J[ii].T * self._M[ii] * J[ii]
            Mq = sp.Matrix(Mq)

            # save to file
            cloudpickle.dump(Mq, open('%s/Mq' % self.config_folder, 'wb'))

        if lambdify is False:
            return Mq
        return sp.lambdify(self.q + self.x, Mq)

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
            J = [self._calc_J('link%s' % ii, x=[0, 0, 0], lambdify=False)
                 for ii in range(self.num_links)]

            # transform each inertia matrix into joint space and
            # sum together the effects of arm segments' inertia on each motor
            Mq_g = sp.zeros(self.num_joints, 1)
            for ii in range(self.num_joints):
                Mq_g += J[ii].T * self._M[ii] * self.gravity
            Mq_g = sp.Matrix(Mq_g)

            # save to file
            cloudpickle.dump(Mq_g, open('%s/Mq_g' % self.config_folder,
                                        'wb'))

        if lambdify is False:
            return Mq_g
        return sp.lambdify(self.q + self.x, Mq_g)

    def _calc_T(self, name):  # noqa C907
        """ Uses Sympy to generate the transform for a joint or link

        name string: name of the joint or link, or end-effector
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        if name == 'joint0' or name == 'link0':
            T = self.Torg0
        elif name == 'joint1' or name == 'link1':
            T = self.Torg0 * self.T01
        elif name == 'joint2':
            T = self.Torg0 * self.T01 * self.T12
        elif name == 'link2':
            T = self.Torg0 * self.T01 * self.T1l2
        elif name == 'joint3':
            T = self.Torg0 * self.T01 * self.T12 * self.T23
        elif name == 'link3':
            T = self.Torg0 * self.T01 * self.T12 * self.T2l3
        elif name == 'joint4' or name == 'link4':
            T = self.Torg0 * self.T01 * self.T12 * self.T23 * self.T34
        elif name == 'joint5' or name == 'link5':
            T = self.Torg0 * self.T01 * self.T12 * self.T23 * self.T34 * \
                self.T45
        elif name == 'link6' or name == 'EE':
            T = self.Torg0 * self.T01 * self.T12 * self.T23 * self.T34 * \
                self.T45 * self.T5EE
        else:
            raise Exception('Invalid transformation name: %s' % name)
        return T

    def _calc_Tx(self, name, x, lambdify=True):
        """ Uses Sympy to transform x from the reference frame of a joint
        or link to the origin (world) coordinates.

        name string: name of the joint or link, or end-effector
        x list: the [x,y,z] position of interest in "name"'s reference frame
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        if (os.path.isfile('%s/%s.T' % (self.config_folder, name))):
            Tx = cloudpickle.load(open('%s/%s.T' %
                                       (self.config_folder, name), 'rb'))
        else:
            T = self._calc_T(name=name)
            # transform x into world coordinates
            Tx = T * sp.Matrix(self.x + [1])

            # save to file
            cloudpickle.dump(Tx, open('%s/%s.T' %
                                      (self.config_folder, name), 'wb'))

        if lambdify is False:
            return Tx
        return sp.lambdify(self.q + self.x, Tx)

    def _calc_T_inv(self, name, x, lambdify=True):
        """ Return the inverse transform matrix, which converts from
        world coordinates into the robot's end-effector reference frame

        name string: name of the joint or link, or end-effector
        x list: the [x,y,z] position of interest in "name"'s reference frame
        lambdify boolean: if True returns a function to calculate
                          the transform. If False returns the Sympy
                          matrix
        """

        # check to see if we have our transformation saved in file
        if (os.path.isfile('%s/%s.T_inv' % (self.config_folder,
                                                name))):
            T_inv = cloudpickle.load(open('%s/%s.T_inv' %
                                          (self.config_folder, name), 'rb'))
        else:
            T = self._calc_T(name=name)
            rotation_inv = T[:3, :3].T
            translation_inv = -rotation_inv * T[:3, 3]
            T_inv = rotation_inv.row_join(translation_inv).col_join(
                sp.Matrix([[0, 0, 0, 1]]))

            # save to file
            cloudpickle.dump(T_inv, open('%s/%s.T_inv' %
                                         (self.config_folder, name), 'wb'))

        if lambdify is False:
            return T_inv
        return sp.lambdify(self.q + self.x, T_inv)
