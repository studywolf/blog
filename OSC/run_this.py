from Arms.TwoLinkArm.arm import Arm
from Controllers.TwoLinkArm.control import Control
'''from Arms.OneLinkArm.arm import Arm
from Controllers.OneLinkArm.control import Control'''
from runner import Runner
import numpy as np

# One link arm control_type = 'gc', 'osc_x', 'osc_y'
# Two link arm control_type = 'gc', 'osc'
# xylim controls the size of the figure

# Two link arm can also run pure python by importing 
#   from Arms.TwoLinkArm.arm_python import Arm

kp = 10 # gain on the PD controller
arm = Arm()
control = Control(kp=kp, kv=np.sqrt(kp), control_type='osc_x')
runner = Runner(title='2LinkArm', dt=1e-4, control_steps=10, 
                display_steps=100, t_target=2., max_tau=1e100,
                xylim=1)

runner.run(arm=arm, control=control)
runner.show()
