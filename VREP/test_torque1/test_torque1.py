import numpy as np
import vrep
import time

# close any open connections
vrep.simxFinish(-1) 
# Connect to V-REP
clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5) 

if clientID != -1: # if we connected successfully 
    print ('Connected to remote API server')

    # Now try to retrieve data in a blocking fashion (i.e. a service call):
    res, objs = vrep.simxGetObjects(clientID, 
                                    vrep.sim_handle_all, 
                                    vrep.simx_opmode_blocking)
    if res != vrep.simx_return_ok:
        raise Exception('Remote API function call returned with error code: ',res)

    vrep.simxSynchronous(clientID,True)
    vrep.simxAddStatusbarMessage(clientID, 'poop', 
            vrep.simx_opmode_oneshot)

    joint_names = ['joint0', 'joint1']
    joint_handles = []
    joint_velocities = {}

    # get the handles for each joint 
    for ii,name in enumerate(joint_names):
        returnCode, joint_handle = vrep.simxGetObjectHandle(clientID,
                name, vrep.simx_opmode_blocking) 
        joint_handles.append(joint_handle)
        print 'returnCode: ,', returnCode
        print '%s handle: %i'%(name, joint_handle)

        # initialize the data collection from the joints
        vrep.simxGetJointForce(clientID,
                joint_handle,
                vrep.simx_opmode_streaming)
        # set the target velocities of each joint super high
        # and then we'll control the max torque allowed (yeah, i know)
        joint_velocities[joint_handle] = 50.0
        vrep.simxSetJointTargetVelocity(clientID,
                joint_handle,
                joint_velocities[joint_handle], # target velocity
                vrep.simx_opmode_oneshot)

    # After initialization of streaming, it will take a few ms before the 
    # first value arrives, so check the return code
    time.sleep(.1)

    count = 0
    start_time = time.time()
    while time.time() - start_time < 5:
        for joint_handle in joint_handles:
            # Try to retrieve the streamed data
            returnCode, data = \
                vrep.simxGetJointForce(clientID,
                        joint_handle,
                        vrep.simx_opmode_buffer) 
            
            print ('Joint %i torque is : %.3f'%(joint_handle, data))

            # calculate force to apply 
            force = np.sin(count) * 10

            # if force has changed signs, 
            # we need to change the target velocity sign
            if np.sign(data) * np.sign(force) < 0:
                print 'changing sign'
                joint_velocities[joint_handle] = joint_velocities[joint_handle] * -1
                print joint_velocities[joint_handle]
                vrep.simxSetJointTargetVelocity(clientID,
                        joint_handle,
                        joint_velocities[joint_handle], # target velocity
                        vrep.simx_opmode_oneshot)
            
            # and now modulate the force
            vrep.simxSetJointForce(clientID, 
                    joint_handle,
                    abs(force), # force to apply
                    vrep.simx_opmode_blocking)

        count += .1
        time.sleep(0.005)

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(clientID, 'Hello V-REP!',
                                 vrep.simx_opmode_oneshot)

    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # Before closing the connection to V-REP, 
    #make sure that the last command sent out had time to arrive. 
    vrep.simxGetPingTime(clientID)

    # Now close the connection to V-REP:
    vrep.simxFinish(clientID)
else:
    print ('Failed connecting to remote API server')
print ('Program ended')
