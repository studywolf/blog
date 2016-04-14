import numpy as np
import vrep
import time

# close any open connections
vrep.simxFinish(-1) 
# Connect to the V-REP continuous server
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 500, 5) 

if clientID != -1: # if we connected successfully 
    print ('Connected to remote API server')

    # --------------------- Setup the simulation 

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs = vrep.simxGetObjects(clientID, 
                                    vrep.sim_handle_all, 
                                    vrep.simx_opmode_blocking)
    if res != vrep.simx_return_ok:
        raise Exception('Remote API function call returned with error code: ',res)

    vrep.simxSynchronous(clientID,True)

    joint_names = ['joint0', 'joint1']
    cube_names = ['upper_arm', 'forearm', 'hand']
    joint_handles = []
    cube_handles = []
    joint_angles = {}
    joint_velocities = {}
    joint_target_velocities = {}
    joint_forces = {}

    # get the handles for each joint and set up streaming
    for ii,name in enumerate(joint_names):
        _, joint_handle = vrep.simxGetObjectHandle(clientID,
                name, vrep.simx_opmode_blocking) 
        joint_handles.append(joint_handle)
        print '%s handle: %i'%(name, joint_handle)

        # initialize the data collection from the joints
        vrep.simxGetJointForce(clientID,
                joint_handle,
                vrep.simx_opmode_streaming)
        vrep.simxGetJointPosition(clientID,
                joint_handle,
                vrep.simx_opmode_streaming)
        vrep.simxGetObjectFloatParameter(clientID,
                joint_handle,
                2012, # parameter ID for angular velocity we want
                vrep.simx_opmode_streaming)
        # set the target velocities of each joint super high
        # and then we'll control the max torque allowed (yeah, i know)
        joint_target_velocities[joint_handle] = 100.0
        vrep.simxSetJointTargetVelocity(clientID,
                joint_handle,
                joint_target_velocities[joint_handle], # target velocity
                vrep.simx_opmode_oneshot)

    # get the handle for our cubes and set up streaming
    for name in cube_names:
        _, handle = vrep.simxGetObjectHandle(clientID,
                    name, vrep.simx_opmode_blocking) 
        cube_handles.append(handle)
        # start streaming the (x,y,z) position of the cubes
        vrep.simxGetObjectPosition(clientID,
                handle, 
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_streaming)

    # --------------------- Run the simulation
    vrep.simxSetFloatingParameter(clientID, 
            vrep.sim_floatparam_simulation_time_step, 
            .001, # specify a simulation time step
            vrep.simx_opmode_oneshot)
    # start our simulation in lockstep with our code
    vrep.simxStartSimulation(clientID,
            vrep.simx_opmode_blocking)

    # After initialization of streaming, it will take a few ms before the 
    # first value arrives, so check the return code
    time.sleep(.1)

    target_xyz = np.array([-0.1118, -0.5500, 0.6259])
    count = 0
    track_hand = []
    start_time = time.time()
    while time.time() - start_time < 5:
        # move simulation ahead one
        vrep.simxSynchronousTrigger(clientID)

        # get the (x,y,z) position of the hand
        _, xyz = vrep.simxGetObjectPosition(clientID,
                cube_handles[-1], 
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_buffer)
        track_hand.append(np.copy(xyz))

        # calculate desired movement in operational (hand) space 
        kp = 1500.0
        kv = -2.5#np.sqrt(kp)
        u_xyz = kp * (target_xyz - xyz)

        for joint_handle in joint_handles: 
            # get the joint angles 
            _, joint_angle = vrep.simxGetJointPosition(clientID,
                    joint_handle,
                    vrep.simx_opmode_buffer)
            joint_angles[joint_handle] = joint_angle
            _, joint_velocity = vrep.simxGetObjectFloatParameter(clientID,
                    joint_handle,
                    2012, # parameter ID for angular velocity of the joint
                    vrep.simx_opmode_buffer)
            joint_velocities[joint_handle] = joint_velocity

        # calculate the Jacobian
        q0 = joint_angles[joint_handles[0]]
        q1 = joint_angles[joint_handles[1]]
        J = np.zeros((2,2))
        J[0][0] = -.4*np.sin(q0)
        J[0][1] = J[0][0] - .2*np.sin(q0+q1)
        J[1][0] = .4*np.cos(q0)
        J[1][1] = J[1][0] + .2*np.cos(q0+q1)

        u_xz = u_xyz[[0,2]]
        print u_xz / kp 
        u = (np.dot(J.T, u_xz) - 
                kv * np.array([joint_velocities[joint_handles[0]],
                               joint_velocities[joint_handles[1]]]))

        joint_forces[joint_handles[0]] = u[0]
        joint_forces[joint_handles[1]] = u[1]

        for joint_handle in joint_handles:

            # get the current joint torque
            _, torque = \
                vrep.simxGetJointForce(clientID,
                        joint_handle,
                        vrep.simx_opmode_buffer) 

            # if force has changed signs, 
            # we need to change the target velocity sign
            if np.sign(torque) * np.sign(joint_forces[joint_handle]) < 0:
                joint_target_velocities[joint_handle] = \
                        joint_target_velocities[joint_handle] * -1
                vrep.simxSetJointTargetVelocity(clientID,
                        joint_handle,
                        joint_target_velocities[joint_handle], # target velocity
                        vrep.simx_opmode_oneshot)
            
            # and now modulate the force
            vrep.simxSetJointForce(clientID, 
                    joint_handle,
                    abs(joint_forces[joint_handle]), # force to apply
                    vrep.simx_opmode_blocking)

        count += .1

    # stop the simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # Before closing the connection to V-REP, 
    #make sure that the last command sent out had time to arrive. 
    vrep.simxGetPingTime(clientID)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
else:
    raise Exception('Failed connecting to remote API server')

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

track_hand = np.array(track_hand)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([track_hand[0,0]], [track_hand[0,1]], [track_hand[0,2]], 'bx', mew=10)
ax.plot(track_hand[:,0], track_hand[:,1], track_hand[:,2])
ax.plot([target_xyz[0]], [target_xyz[1]], [target_xyz[2]], 'rx', mew=10)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 0])
ax.set_zlim([0, 1])
ax.legend()

plt.show()
