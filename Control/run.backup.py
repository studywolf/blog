'''
Copyright (C) 2014 Travis DeWolf

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

from sim_and_plot import Runner
import numpy as np

arm = 1
control_type = 'dmp'# 'adaptive_osc' 

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
    #   from Arms.TwoLinkArm.arm_python import Arm
    from Arms.TwoLinkArm.arm import Arm2Link as Arm
    arm_pars['singularity_thresh'] = 1e-5
    # Two link arm control_type = 'gc', 'osc', 'dmp', 'trajectory'
    runner_pars = {'control_type':control_type, 
                   'title':'2 link arm',
                   'box':[-.75, .75, -.75, .75]}

elif arm == 3: 
    from Arms.ThreeLinkArm.arm import Arm3Link as Arm
    # Three link arm control_type = 'gc', 'osc', 'osc_and_null'
    runner_pars = {'control_type':control_type,
                   'title':'3 link arm',
                   'box':[-4, 4, -4, 4]}

#--------------------------------
# set up the chosen controller

control_pars = {}

if control_type == 'gc':
    # generalized coordinates control
    from Controllers.control_GC import Control_GC as Control

elif control_type[-3:] == 'osc':
    if arm == 1: 
        raise Exception('invalid control type for single link arm')
    # operational space control in (x, y) space

    # null control adds another controller in the null space
    # that attempts to keep the arm near resting state joint angles 
    control_pars['null_control'] = False#True
    
    if control_type == 'osc':
        from Controllers.control_OSC import Control_OSC as Control
    else:
        from Controllers.control_adaptive_OSC \
                import Control_Adaptive_OSC as Control

elif control_type.startswith(('dmp', 'trajectory')):
    # trajectory following or dynamic movement primitive based control

    # area to scale trajectory to
    xbias = 0
    ybias = 0
    writebox = np.ones(4)*2 + np.array([x_bias, x_bias, y_bias, y_bias])

    if control_type.endswith('osc'):
        # ----for writing numbers----
        #import Controllers.Trajectories.Writing.number_array as na
        #trajectory = na.get_sequence([3,3,3], writebox)
        # ---------------------------

        # ----for writing words------
        import Controllers.Trajectories.Writing.read_trajectory as rt
        #trajectory = rt.read_file('Controllers/Trajectories/Writing/ca0.dat', 
                                  #'', box=writebox)
        trajectory = rt.read_file_pkl('Controllers/Trajectories/Writing/pen_pos.pkl',
                                      '', box=writebox)#[:210]
        # ---------------------------
    elif control_type.endswith('gc'):
        import Controllers.
    control_pars['trajectory'] = trajectory

    if control_type == 'dmp':
        from Controllers.control_DMP import Control_DMP as Control
        control_pars.update({'bfs':1000, # how many basis function per DMP
                             'tau':.01, # tau is the time scaling term
                             'add_to_goals':[0,0, 0,-.3, 5e-4,-.3]}) # respecify goals for spatial scaling
    else: 
        from Controllers.control_trajectory import Control_trajectory as Control
        control_pars.update({'dt':.00001}) # how fast the trajectories rolls out

    control_pars.update({'pen_down':False, 
                         'gain':1000, # pd gain for trajectory following
                         'trajectory':trajectory.T})
    #[0, 1.5, 1, 3]
    runner_pars.update({'box':[-1.75, 1.75, 1, 2.5],#[-3,3,0,3],#[-.5, .5, 0, .5],
                       'trajectory':trajectory})

#--------------------------------
# run and plot the system

kp = 5 # position error gain on the PD controller

arm = Arm(**arm_pars)
control = Control(kp=kp, kv=np.sqrt(kp), **control_pars)

# set up mouse control
runner = Runner(dt=1e-4, 
                control_steps=1,
                display_steps=100, 
                t_target=1., 
                **runner_pars)

runner.run(arm=arm, control=control)#, video='word.mp4')
runner.show()
