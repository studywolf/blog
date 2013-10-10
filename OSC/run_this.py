#from TwoLinkArm.arm import Arm # uncomment this to use MapleSim arm sim
from TwoLinkArm.arm_python import Arm
from TwoLinkArm.control import Control
from runner import Runner
import numpy as np

kp = 10
arm = Arm()
control = Control(kp=kp, kv=np.sqrt(kp), control_type='osc')
runner = Runner(title='2LinkArm', dt=1e-4, control_steps=1, 
                display_steps=100, t_target=.75, max_tau=1e100,
                xylim=.7)

runner.run(arm=arm, control=control)
runner.show()
