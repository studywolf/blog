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

    joint_names = ['shoulder', 'elbow']
    joint_handles = []
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
        joint_target_velocities[joint_handle] = 10000.0
        vrep.simxSetJointTargetVelocity(clientID,
                joint_handle,
                joint_target_velocities[joint_handle], # target velocity
                vrep.simx_opmode_oneshot)

    # get handle for target and set up streaming
    _, target_handle = vrep.simxGetObjectHandle(clientID,
                    'target', vrep.simx_opmode_blocking) 
    _, target_xyz = vrep.simxGetObjectPosition(clientID,
                target_handle, 
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_streaming)



    # --------------------- Run the simulation
    dt = .001
    vrep.simxSetFloatingParameter(clientID, 
            vrep.sim_floatparam_simulation_time_step, 
            dt, # specify a simulation time step
            vrep.simx_opmode_oneshot)
    # start our simulation in lockstep with our code
    vrep.simxStartSimulation(clientID,
            vrep.simx_opmode_blocking)

    # After initialization of streaming, it will take a few ms before the 
    # first value arrives, so check the return code
    time.sleep(.1)

    count = 0
    track_hand = []
    track_target = []
    start_time = time.time()
    # while time.time() - start_time < 10:
    while count < .1:
        
        # get the (x,y,z) position of the target
        _, target_xyz = vrep.simxGetObjectPosition(clientID,
                target_handle, 
                -1, # retrieve absolute, not relative, position
                vrep.simx_opmode_blocking)
        if _ !=0 : raise Exception()
        track_target.append(np.copy(target_xyz))
        target_xyz = np.asarray(target_xyz)

        for joint_handle in joint_handles: 
            # get the joint angles 
            _, joint_angle = vrep.simxGetJointPosition(clientID,
                    joint_handle,
                    vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            joint_angles[joint_handle] = joint_angle
            _, joint_velocity = vrep.simxGetObjectFloatParameter(clientID,
                    joint_handle,
                    2012, # parameter ID for angular velocity of the joint
                    vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            joint_velocities[joint_handle] = joint_velocity
        dq = np.array([joint_velocities[joint_handles[0]],
                       joint_velocities[joint_handles[1]]])

        q = np.array([joint_angles[joint_handles[0]],
                      joint_angles[joint_handles[1]]])
        # note that .15 is the distance to the center of 
        # the hand, which is the (x,y,z) returned from VREP
        L = np.array([.42, .25]) # arm segment lengths

        xyz = np.array([L[0] * np.cos(q[0]) + L[1] * np.cos(q[0]+q[1]),
                        0, 
                        L[0] * np.sin(q[0]) + L[1] * np.sin(q[0]+q[1]) + .1])
        track_hand.append(np.copy(xyz))
        # calculate the Jacobian for the hand
        JEE = np.zeros((3,2))
        JEE[0,1] = L[1] * -np.sin(q[0]+q[1])
        JEE[2,1] = L[1] * np.cos(q[0]+q[1]) 
        JEE[0,0] = L[0] * -np.sin(q[0]) + JEE[0,1]
        JEE[2,0] = L[0] * np.cos(q[0]) + JEE[2,1]

        # get the Jacobians for the centres-of-mass for the arm segments 
        JCOM1 = np.zeros((6,2))
        JCOM1[0,0] = .22 * -np.sin(q[0]) # COM is in a weird place 
        JCOM1[2,0] = .22 * np.cos(q[0]) # because of offset
        JCOM1[4,0] = 1.0

        JCOM2 = np.zeros((6,2))
        JCOM2[0,1] = .15 * -np.sin(q[0]+q[1]) # COM is in a weird place 
        JCOM2[2,1] = .15 * np.cos(q[0]+q[1]) # because of offset
        JCOM2[4,1] = 1.0
        JCOM2[0,0] = L[0] * -np.sin(q[0]) + JCOM2[0,1]
        JCOM2[2,0] = L[0] * np.cos(q[0]) + JCOM2[2,1]
        JCOM2[4,0] = 1.0

        m1 = 1.171e-0 # from VREP
        i1 = 8.121e-3 # from VREP
        M1 = np.diag([m1, m1, m1, i1, i1, 1.242e-3])
        m2 = 7.804e-1 # from VREP
        i2 = 3.954e-3 # from VREP
        M2 = np.diag([m2, m2, m2, i2, i2, 1.242e-3])

        # generate the mass matrix in joint space
        Mq = np.dot(JCOM1.T, np.dot(M1, JCOM1)) + \
             np.dot(JCOM2.T, np.dot(M2, JCOM2))

        # # TODO: why doesn't this work? 
        # Mx_inv = np.dot(JEE, np.dot(np.linalg.inv(Mq), JEE.T))
        # Mu,Ms,Mv = np.linalg.svd(Mx_inv)
        # # cut off any singular values that could cause control problems
        # for i in range(len(Ms)):
        #     Ms[i] = 0 if Ms[i] < .00025 else 1./float(Ms[i])
        # Mx_inv = np.dot(Mv, np.dot(np.diag(Ms), Mu.T))

        # calculate desired movement in operational (hand) space 
        kp = 1000
        kv = np.sqrt(kp)
        u_xyz = kp * (target_xyz - xyz)
        # u_xyz = np.dot(Mx_inv, kp * (target_xyz - xyz))

        u = np.dot(JEE.T, u_xyz) - np.dot(Mq, kv * dq)
        # u = np.dot(JEE.T, np.dot(Mx, u_xyz)) - np.dot(Mq, kv * dq)
        print 'u : ', u

        joint_forces[joint_handles[0]] = -u[0]
        joint_forces[joint_handles[1]] = -u[1]

        for joint_handle in joint_handles:

            # get the current joint torque
            _, torque = \
                vrep.simxGetJointForce(clientID,
                        joint_handle,
                        vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()

            # if force has changed signs, 
            # we need to change the target velocity sign
            if np.sign(torque) * np.sign(joint_forces[joint_handle]) < 0:
                joint_target_velocities[joint_handle] = \
                        joint_target_velocities[joint_handle] * -1
                vrep.simxSetJointTargetVelocity(clientID,
                        joint_handle,
                        joint_target_velocities[joint_handle], # target velocity
                        vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()
            
            # and now modulate the force
            vrep.simxSetJointForce(clientID, 
                    joint_handle,
                    abs(joint_forces[joint_handle]), # force to apply
                    vrep.simx_opmode_blocking)
            if _ !=0 : raise Exception()

        # raw_input()
        # move simulation ahead one
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
