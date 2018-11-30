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
robot_config = arm.Config()

# create our interface
interface = VREP(robot_config, dt=.005)
interface.connect()

# control (x, y, beta, gamma) out of [x, y, z, alpha, beta, gamma]
# NOTE: needs to be an array to properly select elements of J and u_task
ctrlr_dof = np.array([False, False, False, True, True, True])

# control gains
kp = 500
ko = 200
kv = np.sqrt(kp)

try:
    print('\nSimulation starting...\n')
    while 1:
        # get arm feedback
        feedback = interface.get_feedback()
        hand_xyz = robot_config.Tx('EE', feedback['q'])

        # set the block to be the same orientation as end-effector
        rc_matrix = robot_config.R('EE', feedback['q'])
        rc_angles = transformations.euler_from_matrix(
            rc_matrix, axes='rxyz')
        interface.set_orientation('object', rc_angles)

        # get Jacobian and remove uncontrolled dimensions
        J = robot_config.J('EE', q=feedback['q'])[ctrlr_dof]

        # calculate the inertia matrix in task space
        M = robot_config.M(q=feedback['q'])
        Mx_inv = np.dot(J, np.dot(np.linalg.inv(M), J.T))
        Mx = np.linalg.pinv(Mx_inv, rcond=.005)

        target = np.hstack([
            interface.get_xyz('target'),
            interface.get_orientation('target')])

        u_task = np.zeros(6)  # [x, y, z, alpha, beta, gamma]
        # calculate position error
        u_task[:3] = -kp * (hand_xyz - target[:3])

        # calculate orientation error
        q_target = transformations.quaternion_from_euler(
            target[3], target[4], target[5], axes='rxyz')
        q_ee = transformations.quaternion_from_matrix(
            robot_config.R('EE', q=feedback['q']))
        q_r = transformations.quaternion_multiply(
            q_target, transformations.quaternion_conjugate(q_ee))
        u_task[3:] = ko * q_r[1:] * np.sign(q_r[0])

        # remove uncontrolled dimensions from u_task
        u_task = u_task[ctrlr_dof]
        # transform from operational space to torques
        # add in velocity and gravity compensation in joint space
        u = (np.dot(J.T, np.dot(Mx, u_task)) -
                kv * np.dot(M, feedback['dq']) -
                robot_config.g(q=feedback['q']))
        # apply the control signal, step the sim forward
        interface.send_forces(u)

finally:
    interface.disconnect()
