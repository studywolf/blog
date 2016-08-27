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
import vrep

import ur5

# create instance of ur5 class which provides all
# the transform and Jacobian information for this arm
robot_config = ur5.robot_config()

# close any open connections
vrep.simxFinish(-1)
# Connect to the V-REP continuous server
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 500, 5)

try:
    if clientID != -1: # if we connected successfully  # noqa C901
        print('Connected to remote API server')

        # --------------------- Setup the simulation

        vrep.simxSynchronous(clientID, True)

        # joint target velocities discussed below
        joint_target_velocities = np.ones(robot_config.num_joints) * 10000.0

        # get the handles for each joint and set up streaming
        joint_handles = [vrep.simxGetObjectHandle(
            clientID,
            name,
            vrep.simx_opmode_blocking)[1] for name in robot_config.joint_names]

        # get handle for target and set up streaming
        _, target_handle = vrep.simxGetObjectHandle(
            clientID,
            'target',
            vrep.simx_opmode_blocking)
        # get handle for hand
        _, hand_handle = vrep.simxGetObjectHandle(
            clientID,
            'hand',
            vrep.simx_opmode_blocking)

        # define a set of targets
        center = np.array([0, 0, 0.6])
        dist = .2
        num_targets = 10
        target_positions = np.array([
            [dist*np.cos(theta)*np.sin(theta),
             dist*np.cos(theta),
             dist*np.sin(theta)] +
            center for theta in np.linspace(0, np.pi*2, num_targets)])

        # define variables to share with nengo
        q = np.zeros(len(joint_handles))
        dq = np.zeros(len(joint_handles))

        # --------------------- Start the simulation

        dt = .001
        vrep.simxSetFloatingParameter(
            clientID,
            vrep.sim_floatparam_simulation_time_step,
            dt,  # specify a simulation time step
            vrep.simx_opmode_oneshot)

        # start our simulation in lockstep with our code
        vrep.simxStartSimulation(
            clientID,
            vrep.simx_opmode_blocking)

        count = 0
        track_hand = []
        track_target = []
        target_index = 0
        change_target_time = dt*300

        # NOTE: main loop starts here -----------------------------------------
        while count < len(target_positions) * change_target_time:

            # every so often move the target
            if (count % change_target_time) < dt:
                vrep.simxSetObjectPosition(
                    clientID,
                    target_handle,
                    -1,  # set absolute, not relative position
                    target_positions[target_index],
                    vrep.simx_opmode_blocking)
                target_index += 1

            # get the (x,y,z) position of the target
            _, target_xyz = vrep.simxGetObjectPosition(
                clientID,
                target_handle,
                -1,  # retrieve absolute, not relative, position
                vrep.simx_opmode_blocking)
            if _ != 0:
                raise Exception()
            track_target.append(np.copy(target_xyz))  # store for plotting
            target_xyz = np.asarray(target_xyz)

            for ii, joint_handle in enumerate(joint_handles):
                old_q = np.copy(q)
                # get the joint angles
                _, q[ii] = vrep.simxGetJointPosition(
                    clientID,
                    joint_handle,
                    vrep.simx_opmode_blocking)
                if _ != 0:
                    raise Exception()

                # get the joint velocity
                _, dq[ii] = vrep.simxGetObjectFloatParameter(
                    clientID,
                    joint_handle,
                    2012,  # parameter ID for angular velocity of the joint
                    vrep.simx_opmode_blocking)
                if _ != 0:
                    raise Exception()

            # calculate position of the end-effector
            # derived in the ur5 calc_TnJ class
            xyz = robot_config.T('EE', q)

            # calculate the Jacobian for the end effector
            JEE = robot_config.J('EE', q)

            # calculate the inertia matrix in joint space
            Mq = robot_config.Mq(q)

            # calculate the effect of gravity in joint space
            Mq_g = robot_config.Mq_g(q)

            # convert the mass compensation into end effector space
            Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
            svd_u, svd_s, svd_v = np.linalg.svd(Mx_inv)
            # cut off any singular values that could cause control problems
            singularity_thresh = .00025
            for i in range(len(svd_s)):
                svd_s[i] = 0 if svd_s[i] < singularity_thresh else \
                    1./float(svd_s[i])
            # numpy returns U,S,V.T, so have to transpose both here
            Mx = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

            kp = 100
            kv = np.sqrt(kp)
            # calculate desired force in (x,y,z) space
            u_xyz = np.dot(Mx, target_xyz - xyz)
            # transform into joint space, add vel and gravity compensation
            u = (kp * np.dot(JEE.T, u_xyz) - np.dot(Mq, kv * dq) - Mq_g)

            # calculate our secondary control signal
            # calculated desired joint angle acceleration
            q_des = (((robot_config.rest_angles - q) + np.pi) %
                     (np.pi*2) - np.pi)
            u_null = np.dot(Mq, (kp * q_des - kv * dq))

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
            null_filter = (np.eye(robot_config.num_joints) -
                           np.dot(JEE.T, Jdyn_inv))
            u_null_filtered = np.dot(null_filter, u_null)

            u += u_null_filtered

            # multiply by -1 because torque is opposite of expected
            u *= -1
            print('u: ', u)

            for ii, joint_handle in enumerate(joint_handles):
                # the way we're going to do force control is by setting
                # the target velocities of each joint super high and then
                # controlling the max torque allowed (yeah, i know)

                # get the current joint torque
                _, torque = \
                    vrep.simxGetJointForce(
                        clientID,
                        joint_handle,
                        vrep.simx_opmode_blocking)
                if _ != 0:
                    raise Exception()

                # if force has changed signs,
                # we need to change the target velocity sign
                if np.sign(torque) * np.sign(u[ii]) <= 0:
                    joint_target_velocities[ii] = \
                        joint_target_velocities[ii] * -1
                    vrep.simxSetJointTargetVelocity(
                        clientID,
                        joint_handle,
                        joint_target_velocities[ii],  # target velocity
                        vrep.simx_opmode_blocking)
                if _ != 0:
                    raise Exception()

                # and now modulate the force
                vrep.simxSetJointForce(
                    clientID,
                    joint_handle,
                    abs(u[ii]),  # force to apply
                    vrep.simx_opmode_blocking)
                if _ != 0:
                    raise Exception()

            # Update position of hand sphere
            vrep.simxSetObjectPosition(
                clientID,
                hand_handle,
                -1,  # set absolute, not relative position
                xyz,
                vrep.simx_opmode_blocking)
            track_hand.append(np.copy(xyz))  # and store for plotting

            # move simulation ahead one time step
            vrep.simxSynchronousTrigger(clientID)
            count += dt
    else:
        raise Exception('Failed connecting to remote API server')
finally:
    # stop the simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # Before closing the connection to V-REP,
    # make sure that the last command sent out had time to arrive.
    vrep.simxGetPingTime(clientID)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
    print('connection closed...')

    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    track_hand = np.array(track_hand)
    track_target = np.array(track_target)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot start point of hand
    ax.plot([track_hand[0, 0]],
            [track_hand[0, 1]],
            [track_hand[0, 2]],
            'bx', mew=10)
    # plot trajectory of hand
    ax.plot(track_hand[:, 0],
            track_hand[:, 1],
            track_hand[:, 2])
    # plot trajectory of target
    ax.plot(track_target[:, 0],
            track_target[:, 1],
            track_target[:, 2],
            'rx', mew=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([0, 1])
    ax.legend()

    plt.show()
