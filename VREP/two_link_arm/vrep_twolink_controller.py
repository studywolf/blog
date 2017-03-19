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

try: 
    # close any open connections
    vrep.simxFinish(-1)
    # Connect to the V-REP continuous server
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 500, 5)

    if clientID != -1: # if we connected successfully
        print ('Connected to remote API server')

        # --------------------- Setup the simulation

        vrep.simxSynchronous(clientID,True)

        joint_names = ['shoulder', 'elbow']
        # joint target velocities discussed below
        joint_target_velocities = np.ones(len(joint_names)) * 10000.0

        # get the handles for each joint and set up streaming
        joint_handles = [vrep.simxGetObjectHandle(clientID,
            name, vrep.simx_opmode_blocking)[1] for name in joint_names]

        # get handle for target and set up streaming
        _, target_handle = vrep.simxGetObjectHandle(clientID,
                        'target', vrep.simx_opmode_blocking)

        dt = .001
        vrep.simxSetFloatingParameter(clientID,
                vrep.sim_floatparam_simulation_time_step,
                dt, # specify a simulation time step
                vrep.simx_opmode_oneshot)

        # --------------------- Start the simulation

        # start our simulation in lockstep with our code
        vrep.simxStartSimulation(clientID,
                vrep.simx_opmode_blocking)

        count = 0
        track_hand = []
        track_target = []
        while count < 1: # run for 1 simulated second

            # get the (x,y,z) position of the target
            _, target_xyz = vrep.simxGetObjectPosition(clientID,
                    target_handle,
                    -1, # retrieve absolute, not relative, position
                    vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            track_target.append(np.copy(target_xyz)) # store for plotting
            target_xyz = np.asarray(target_xyz)

            q = np.zeros(len(joint_handles))
            dq = np.zeros(len(joint_handles))
            for ii,joint_handle in enumerate(joint_handles):
                # get the joint angles
                _, q[ii] = vrep.simxGetJointPosition(clientID,
                        joint_handle,
                        vrep.simx_opmode_blocking)
                if _ !=0 : raise Exception()
                # get the joint velocity
                _, dq[ii] = vrep.simxGetObjectFloatParameter(clientID,
                        joint_handle,
                        2012, # parameter ID for angular velocity of the joint
                        vrep.simx_opmode_blocking)
                if _ !=0 : raise Exception()

            L = np.array([.42, .225]) # arm segment lengths

            xyz = np.array([L[0] * np.cos(q[0]) + L[1] * np.cos(q[0]+q[1]),
                            0,
                            # have to add .1 offset to z position
                            L[0] * np.sin(q[0]) + L[1] * np.sin(q[0]+q[1]) + .15])
            track_hand.append(np.copy(xyz)) # store for plotting
            # calculate the Jacobian for the hand
            JEE = np.zeros((3,2))
            JEE[0,1] = L[1] * -np.sin(q[0]+q[1])
            JEE[2,1] = L[1] * np.cos(q[0]+q[1])
            JEE[0,0] = L[0] * -np.sin(q[0]) + JEE[0,1]
            JEE[2,0] = L[0] * np.cos(q[0]) + JEE[2,1]

            # get the Jacobians for the centres-of-mass for the arm segments
            JCOM1 = np.zeros((6,2))
            JCOM1[0,0] = .22 * -np.sin(q[0]) # COM is in a weird place
            JCOM1[2,0] = .22 * np.cos(q[0])  # because of offset
            JCOM1[4,0] = 1.0

            JCOM2 = np.zeros((6,2))
            JCOM2[0,1] = .15 * -np.sin(q[0]+q[1]) # COM is in a weird place
            JCOM2[2,1] = .15 * np.cos(q[0]+q[1])  # because of offset
            JCOM2[4,1] = 1.0
            JCOM2[0,0] = L[0] * -np.sin(q[0]) + JCOM2[0,1]
            JCOM2[2,0] = L[0] * np.cos(q[0]) + JCOM2[2,1]
            JCOM2[4,0] = 1.0

            m1 = 1 # from VREP
            i1 = .5 # from VREP
            M1 = np.diag([m1, m1, m1, i1, i1, i1]) * 2
            m2 = 2 # from VREP
            i2 = .1 # from VREP
            M2 = np.diag([m2, m2, m2, i2, i2, i2]) * 2

            # generate the mass matrix in joint space
            Mq = np.dot(JCOM1.T, np.dot(M1, JCOM1)) + \
                 np.dot(JCOM2.T, np.dot(M2, JCOM2))

            # compensate for gravity
            gravity = np.array([0, 0, -9.81, 0, 0, 0,])
            Mq_g = np.dot(JCOM1.T, np.dot(M1, gravity)) + \
                    np.dot(JCOM2.T, np.dot(M2, gravity))

            Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
            Mu,Ms,Mv = np.linalg.svd(Mx_inv)
            # cut off any singular values that could cause control problems
            for i in range(len(Ms)):
                Ms[i] = 0 if Ms[i] < 1e-5 else 1./float(Ms[i])
            # numpy returns U,S,V.T, so have to transpose both here
            Mx = np.dot(Mv.T, np.dot(np.diag(Ms), Mu.T))

            # calculate desired movement in operational (hand) space
            kp = 100
            kv = np.sqrt(kp)
            u_xyz = np.dot(Mx, kp * (target_xyz - xyz))

            u = np.dot(JEE.T, u_xyz) - np.dot(Mq, kv * dq) - Mq_g
            u *= -1 # because the joints on the arm are backwards

            for ii,joint_handle in enumerate(joint_handles):
                # the way we're going to do force control is by setting
                # the target velocities of each joint super high and then
                # controlling the max torque allowed (yeah, i know)

                # get the current joint torque
                _, torque = \
                    vrep.simxGetJointForce(clientID,
                            joint_handle,
                            vrep.simx_opmode_blocking)
                if _ !=0 : raise Exception()

                # if force has changed signs,
                # we need to change the target velocity sign
                if np.sign(torque) * np.sign(u[ii]) < 0:
                    joint_target_velocities[ii] = \
                            joint_target_velocities[ii] * -1
                    vrep.simxSetJointTargetVelocity(clientID,
                            joint_handle,
                            joint_target_velocities[ii], # target velocity
                            vrep.simx_opmode_blocking)
                if _ !=0 : raise Exception()

                # and now modulate the force
                vrep.simxSetJointForce(clientID,
                        joint_handle,
                        abs(u[ii]), # force to apply
                        vrep.simx_opmode_blocking)
                if _ !=0 : raise Exception()

            # move simulation ahead one time step
            vrep.simxSynchronousTrigger(clientID)
            count += dt

        # stop the simulation
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

        # Before closing the connection to V-REP,
        #make sure that the last command sent out had time to arrive.
        vrep.simxGetPingTime(clientID)

        # Now close the connection to V-REP:
        vrep.simxFinish(clientID)
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
    ax.plot([track_hand[0,0]], [track_hand[0,1]], [track_hand[0,2]], 'bx', mew=10)
    # plot trajectory of hand
    ax.plot(track_hand[:,0], track_hand[:,1], track_hand[:,2])
    # plot trajectory of target
    ax.plot(track_target[:,0], track_target[:,1], track_target[:,2], 'rx', mew=10)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-.5, .5])
    ax.set_zlim([0, 1])
    ax.legend()

    plt.show()
