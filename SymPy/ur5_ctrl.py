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
        # get handle for obstacle and set up streaming
        _, obstacle_handle = vrep.simxGetObjectHandle(
            clientID,
            'obstacle',
            vrep.simx_opmode_blocking)

        # how close can we get to the obstacle?
        threshold = .2  # distance in metres
        obstacle_radius = .05

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

        track_hand = []
        track_target = []
        track_obstacle = []

        count = 0
        target_index = 0
        change_target_time = dt*1000
        vmax = 0.5
        kp = 200.0
        kv = np.sqrt(kp)

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
            xyz = robot_config.Tx('EE', q)

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

            # calculate desired force in (x,y,z) space
            dx = np.dot(JEE, dq)
            # implement velocity limiting
            lamb = kp / kv
            x_tilde = xyz - target_xyz
            sat = vmax / (lamb * np.abs(x_tilde))
            scale = np.ones(3)
            if np.any(sat < 1):
                index = np.argmin(sat)
                unclipped = kp * x_tilde[index]
                clipped = kv * vmax * np.sign(x_tilde[index])
                scale = np.ones(3) * clipped / unclipped
                scale[index] = 1
            u_xyz = -kv * (dx - np.clip(sat / scale, 0, 1) *
                                -lamb * scale * x_tilde)
            u_xyz = np.dot(Mx, u_xyz)

            # transform into joint space, add vel and gravity compensation
            u = np.dot(JEE.T, u_xyz) - Mq_g

            # calculate the null space filter
            Jdyn_inv = np.dot(Mx, np.dot(JEE, np.linalg.inv(Mq)))
            null_filter = (np.eye(robot_config.num_joints) -
                           np.dot(JEE.T, Jdyn_inv))
            # calculate our secondary control signal
            q_des = np.zeros(robot_config.num_joints)
            dq_des = np.zeros(robot_config.num_joints)
            # calculated desired joint angle acceleration using rest angles
            for ii in range(1, robot_config.num_joints):
                if robot_config.rest_angles[ii] is not None:
                    q_des[ii] = (
                        ((robot_config.rest_angles[ii] - q[ii]) + np.pi) %
                        (np.pi*2) - np.pi)
                    dq_des[ii] = dq[ii]
            # only compensate for velocity for joints with a control signal
            nkp = kp * .1
            nkv = np.sqrt(nkp)
            u_null = np.dot(Mq, (nkp * q_des - nkv * dq_des))

            u += np.dot(null_filter, u_null)

            # get the (x,y,z) position of the center of the obstacle
            _, v = vrep.simxGetObjectPosition(
                clientID,
                obstacle_handle,
                -1,  # retrieve absolute, not relative, position
                vrep.simx_opmode_blocking)
            if _ != 0:
                raise Exception()
            track_obstacle.append(np.copy(v))  # store for plotting
            v = np.asarray(v)

            # find the closest point of each link to the obstacle
            for ii in range(robot_config.num_joints):
                # get the start and end-points of the arm segment
                p1 = robot_config.Tx('joint%i' % ii, q=q)
                if ii == robot_config.num_joints - 1:
                    p2 = robot_config.Tx('EE', q=q)
                else:
                    p2 = robot_config.Tx('joint%i' % (ii + 1), q=q)

                # calculate minimum distance from arm segment to obstacle
                # the vector of our line
                vec_line = p2 - p1
                # the vector from the obstacle to the first line point
                vec_ob_line = v - p1
                # calculate the projection normalized by length of arm segment
                projection = np.dot(vec_ob_line, vec_line) / np.sum((vec_line)**2)
                if projection < 0:
                    # then closest point is the start of the segment
                    closest = p1
                elif projection > 1:
                    # then closest point is the end of the segment
                    closest = p2
                else:
                    closest = p1 + projection * vec_line
                # calculate distance from obstacle vertex to the closest point
                dist = np.sqrt(np.sum((v - closest)**2))
                # account for size of obstacle
                rho = dist - obstacle_radius

                if rho < threshold:

                    eta = .02  # feel like i saw 4 somewhere in the paper
                    drhodx = (v - closest) / rho
                    Fpsp = (eta * (1.0/rho - 1.0/threshold) *
                            1.0/rho**2 * drhodx)

                    # get offset of closest point from link's reference frame
                    T_inv = robot_config.T_inv('link%i' % ii, q=q)
                    m = np.dot(T_inv, np.hstack([closest, [1]]))[:-1]
                    # calculate the Jacobian for this point
                    Jpsp = robot_config.J('link%i' % ii, x=m, q=q)[:3]

                    # calculate the inertia matrix for the
                    # point subjected to the potential space
                    Mxpsp_inv = np.dot(Jpsp,
                                    np.dot(np.linalg.pinv(Mq), Jpsp.T))
                    svd_u, svd_s, svd_v = np.linalg.svd(Mxpsp_inv)
                    # cut off singular values that could cause problems
                    singularity_thresh = .00025
                    for ii in range(len(svd_s)):
                        svd_s[ii] = 0 if svd_s[ii] < singularity_thresh else \
                            1./float(svd_s[ii])
                    # numpy returns U,S,V.T, so have to transpose both here
                    Mxpsp = np.dot(svd_v.T, np.dot(np.diag(svd_s), svd_u.T))

                    u_psp = -np.dot(Jpsp.T, np.dot(Mxpsp, Fpsp))
                    if rho < .01:
                        u = u_psp
                    else:
                        u += u_psp

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
    track_obstacle = np.array(track_obstacle)

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
    # plot trajectory of obstacle
    ax.plot(track_obstacle[:, 0],
            track_obstacle[:, 1],
            track_obstacle[:, 2],
            'yx', mew=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([0, 1])
    ax.legend()

    plt.show()
