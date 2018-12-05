'''
Copyright (C) 2018 Travis DeWolf

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

from abr_control.arms import ur5 as arm
from abr_control.interfaces import VREP
from abr_control.utils import transformations


# initialize our robot config
robot_config = arm.Config(use_cython=True)

# create our interface
interface = VREP(robot_config, dt=.005)
interface.connect()

# specify which parameters [x, y, z, alpha, beta, gamma] to control
# NOTE: needs to be an array to properly select elements of J and u_task
ctrlr_dof = np.array([True, True, True, True, True, True])

# control gains
kp = 300
ko = 300
kv = np.sqrt(kp+ko) * 1.5

orientations = [
    [0, 0, 0],
    [np.pi/4, np.pi/4, np.pi/4],
    [-np.pi/4, -np.pi/4, np.pi/2],
    [0, 0, 0],
    ]
positions =[
    [0.15, -0.1, 0.6],
    [-.15, 0.0, .7],
    [.2, .2, .6],
    [0.15, -0.1, 0.6]
    ]

try:
    print('\nSimulation starting...\n')
    count = 0
    index = 0
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        if (count % 200) == 0:
            # change target once every simulated second
            if index >= len(orientations):
                break
            interface.set_orientation('target', orientations[index])
            interface.set_xyz('target', positions[index])
            index += 1

        target = np.hstack([
            interface.get_xyz('target'),
            interface.get_orientation('target')])

        # set the block to be the same orientation as end-effector
        R_e = robot_config.R('EE', feedback['q'])
        EA_e = transformations.euler_from_matrix(R_e, axes='rxyz')
        interface.set_orientation('object', EA_e)

        # calculate the Jacobian for the end effectora
        # and isolate relevate dimensions
        J = robot_config.J('EE', q=feedback['q'])[ctrlr_dof]

        # calculate the inertia matrix in task space
        M = robot_config.M(q=feedback['q'])

        # calculate the inertia matrix in task space
        M_inv = np.linalg.inv(M)
        Mx_inv = np.dot(J, np.dot(M_inv, J.T))
        if np.linalg.det(Mx_inv) != 0:
            # do the linalg inverse if matrix is non-singular
            # because it's faster and more accurate
            Mx = np.linalg.inv(Mx_inv)
        else:
            # using the rcond to set singular values < thresh to 0
            # singular values < (rcond * max(singular_values)) set to 0
            Mx = np.linalg.pinv(Mx_inv, rcond=.005)

        u_task = np.zeros(6)  # [x, y, z, alpha, beta, gamma]

        # calculate position error
        u_task[:3] = -kp * (hand_xyz - target[:3])

        # Method 1 ------------------------------------------------------------
        #
        # transform the orientation target into a quaternion
        q_d = transformations.unit_vector(
            transformations.quaternion_from_euler(
                target[3], target[4], target[5], axes='rxyz'))

        # get the quaternion for the end effector
        q_e = transformations.unit_vector(
            transformations.quaternion_from_matrix(
                robot_config.R('EE', feedback['q'])))
        # calculate the rotation from the end-effector to target orientation
        q_r = transformations.quaternion_multiply(
            q_d, transformations.quaternion_conjugate(q_e))
        u_task[3:] = q_r[1:] * np.sign(q_r[0])


        # Method 2 ------------------------------------------------------------
        # From (Caccavale et al, 1997) Section IV - Quaternion feedback -------
        #
        # get rotation matrix for the end effector orientation
        # R_EE = robot_config.R('EE', feedback['q'])
        # # get rotation matrix for the target orientation
        # R_d = transformations.euler_matrix(
        #     target[3], target[4], target[5], axes='rxyz')[:3, :3]
        # R_ed = np.dot(R_EE.T, R_target)  # eq 24
        # q_e = transformations.quaternion_from_matrix(R_ed)
        # q_e = transformations.unit_vector(q_e)
        # u_task[3:] = np.dot(R_EE, q_e[1:])  # eq 34


        # Method 3 ------------------------------------------------------------
        # From (Caccavale et al, 1997) Section V - Angle/axis feedback --------
        #
        # R_EE = robot_config.R('EE', feedback['q'])
        # # get rotation matrix for the target orientation
        # R_d = transformations.euler_matrix(
        #     target[3], target[4], target[5], axes='rxyz')[:3, :3]
        # R = np.dot(R_target, R_EE.T)  # eq 44
        # q_e = transformations.quaternion_from_matrix(R)
        # q_e = transformations.unit_vector(q_e)
        # u_task[3:] = 2 * q_e[0] * q_e[1:]  # eq 45


        # Method 4 -------------------------------------------------------------
        # From (Yuan, 1988) and (Nakanishi et al, 2008) ------------------------
        # NOTE: This implementation is not working properly --------------------
        #
        # # transform the orientation target into a quaternion
        # q_d = transformations.unit_vector(
        #     transformations.quaternion_from_euler(
        #         target[3], target[4], target[5], axes='rxyz'))
        # # get the quaternion for the end effector
        # q_e = transformations.unit_vector(
        #     transformations.quaternion_from_matrix(
        #         robot_config.R('EE', feedback['q'])))
        #
        # # given r = [r1, r2, r3]
        # # r^x = [[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]]
        # S = np.array([
        #     [0.0, -q_d[2], q_d[1]],
        #     [q_d[2], 0.0, -q_d[0]],
        #     [-q_d[1], q_d[0], 0.0]])
        #
        # # calculate the difference between q_e and q_d
        # # from (Nakanishi et al, 2008). NOTE: the sign of the last term
        # # is different from (Yuan, 1998) to account for Euler angles in
        # # world coordinates instead of local coordinates.
        # # dq = (w_d * [x, y, z] - w * [x_d, y_d, z_d] -
        # #       [x_d, y_d, z_d]^x * [x, y, z])
        # # the sign of quaternion that moves between q_e and q_d
        # u_task[3:] = -(q_d[0] * q_e[1:] - q_e[0] * q_d[1:] +
        #                 np.dot(S, q_e[1:]))
        #
        # ---------------------------------------------------------------------

        u_task[3:] *= ko # scale orientation signal by orientation gain
        # remove uncontrolled dimensions from u_task
        u_task = u_task[ctrlr_dof]

        # transform from operational space to torques
        # add in velocity and gravity compensation in joint space
        u = (np.dot(J.T, np.dot(Mx, u_task)) -
                kv * np.dot(M, feedback['dq']) -
                robot_config.g(q=feedback['q']))

        # apply the control signal, step the sim forward
        interface.send_forces(u)
        count += 1

finally:
    interface.disconnect()
