from runner import Runner
import numpy as np

arm = 3 # choose which arm to control

if arm == 1:
    from Arms.OneLinkArm.arm import Arm
    from Controllers.OneLinkArm.control import Control
    # One link arm control_type = 'gc', 'osc_x', 'osc_y'
    # box controls the size of the figure
    control_type = 'osc_x'
    control_pars = {'control_type':control_type}
    runner_pars = {'control_type':control_type, 
                   'title':'1 link arm',
                   'box':[-1, 1, -1, 1]}

elif arm == 2: 
    # Two link arm can also run pure python by importing 
    #   from Arms.TwoLinkArm.arm_python import Arm
    from Arms.TwoLinkArm.arm import Arm
    from Controllers.TwoLinkArm.control import Control
    # Two link arm control_type = 'gc', 'osc'
    control_type = 'osc'
    control_pars = {'control_type':control_type}
    runner_pars = {'control_type':control_type, 
                   'title':'2 link arm',
                   'box':[-.75, .75, -.75, .75]}

elif arm == 3: 
    from Arms.ThreeLinkArm.arm import Arm
    from Controllers.ThreeLinkArm.control import Control
    # Three link arm control_type = 'gc', 'osc', 'osc_and_null'
    control_type = 'osc'
    control_pars = {'control_type':control_type}
    runner_pars = {'control_type':control_type,
                   'title':'3 link arm',
                   'box':[-3, 3, -3, 3]}

kp = 100 # gain on the PD controller
arm = Arm()
control = Control(kp=kp, kv=np.sqrt(kp), **control_pars)

# set up mouse control
runner = Runner(dt=1e-5, 
                control_steps=2,
                display_steps=100, 
                t_target=1., 
                max_tau=1e100,
                **runner_pars)

runner.run(arm=arm, control=control)
runner.show()
