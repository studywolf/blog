from runner import Runner
import numpy as np

# arms: 1, 2, 3
arm = 3
# control_types: gc, osc, dmp, trajectory
control_type = 'dmp' 

#--------------------------------
# set up the chosen arm

arm_pars = {}

if arm == 1:
    from Arms.OneLinkArm.arm import Arm1Link as Arm
    arm_pars['singularity_thresh'] = 1e-10
    # box controls the size of the figure
    runner_pars = {'control_type':control_type, 
                   'title':'1 link arm',
                   'box':[-1, 1, -1, 1]}

elif arm == 2: 
    # Two link arm can also run pure python by importing 
    from Arms.TwoLinkArm.arm_python import Arm2Link as Arm
    #from Arms.TwoLinkArm.arm import Arm2Link as Arm
    arm_pars['singularity_thresh'] = 1e-5
    runner_pars = {'control_type':control_type, 
                   'title':'2 link arm',
                   'box':[-.75, .75, -.75, .75]}

elif arm == 3: 
    from Arms.ThreeLinkArm.arm import Arm3Link as Arm
    runner_pars = {'control_type':control_type,
                   'title':'3 link arm',
                   'box':[-3, 3, -3, 3]}

#--------------------------------
# set up the chosen controller

control_pars = {}

if control_type == 'gc':
    # generalized coordinates control
    from Controllers.control_GC import Control_GC as Control

elif control_type == 'osc':
    if arm == 1: 
        raise Exception('invalid control type for single link arm')
    # operational space control in (x, y) space

    # null control adds another controller in the null space
    # that attempts to keep the arm near resting state joint angles 
    control_pars['null_control'] = True
    from Controllers.control_OSC import Control_OSC as Control

elif control_type in ('trajectory', 'dmp'):
    # trajectory following or dynamic movement primitive based control

    # area to scale trajectory to
    writebox = [-1.5, 1.5, 1.5, 2]

    import Controllers.Trajectories.number_array as na
    trajectory = na.get_sequence(range(10), writebox)
    control_pars['trajectory'] = trajectory

    if control_type == 'dmp':
        from Controllers.control_DMP import Control_DMP as Control
        control_pars.update({'bfs':1000, # how many basis function per DMP
                             'tau':.01}) # tau is the time scaling term
    else: 
        from Controllers.control_trajectory import Control_trajectory as Control
        control_pars.update({'dt':.00001}) # how fast the trajectories rolls out

    control_pars.update({'pen_down':False, 
                         'gain':1000, # pd gain for trajectory following
                         'trajectory':trajectory.T})

    runner_pars.update({'box':[-3,3,0,3], # the viewing area
                       'trajectory':trajectory})

#--------------------------------
# run and plot the system

kp = 500 # position error gain on the PD controller

arm = Arm(**arm_pars)
control = Control(kp=kp, kv=np.sqrt(kp), **control_pars)

# set up mouse control
runner = Runner(dt=1e-4, 
                control_steps=1,
                display_steps=100, 
                t_target=1., 
                max_tau=1e100,
                **runner_pars)

runner.run(arm=arm, control=control)
runner.show()
