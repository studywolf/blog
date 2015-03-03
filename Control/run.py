'''
Copyright (C) 2015 Travis DeWolf

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

Usage:
    run ARM CONTROL TASK [options]

Arguments: 
    ARM     the arm to control
    CONTROL the controller to use 
    TASK    the task to perform

Options:
    --arm_options=OPTIONS   specify options to apply to arm sim 
                            only valid for arm3, choices are:
                            (damping, gravity, gravity_damping, smallmass)
    --use_pygame=PYGAME     specify using pygame for visualization
    --video=VIDPARS  (title, num_frames) [default:('vid.mp4',100)]

'''

from Arms.one_link.arm import Arm1Link as Arm1
from Arms.two_link.arm import Arm2Link as Arm2
from Arms.two_link.arm_python import Arm2Link as Arm2Python
from Arms.three_link.arm import Arm3Link as Arm3

import Controllers.dmp as dmp 
import Controllers.gc as gc 
import Controllers.osc as osc 
import Controllers.trace as trace

import Tasks.follow_mouse as follow_mouse
import Tasks.random_movements as random_movements
import Tasks.reach as reach
import Tasks.write_numbers as write_numbers
import Tasks.write_words as write_words
import Tasks.walk as walk

from docopt import docopt
import numpy as np

args = docopt(__doc__)

# get and instantiate the chosen arm
arm_class = {'arm1':Arm1,
             'arm2':Arm2,
             'arm2_python':Arm2Python,
             'arm3':Arm3}[args['ARM']]
arm = arm_class(options=args['--arm_options'])

# set the initial position of the arm
initial_angles = [np.pi/5.5, np.pi/1.7, np.pi/6.]
initial_state = np.zeros(6)
for d in range(3):
    initial_state[d * 2] = initial_angles[d]
arm.sim.reset(arm.state, initial_state)

# get the chosen controller class
control_class = {'dmp':dmp.Shell,
           'gc':gc.Control,
           'osc':osc.Control,
           'trace':trace.Shell}[args['CONTROL']]

# get the chosen task class
task = {'follow':follow_mouse.Task,
        'random':random_movements.Task,
        'reach':reach.Task,
        'write_numbers':write_numbers.Task,
        'write_words':write_words.Task,
        'walk':walk.Task}[args['TASK']]

# instantiate the controller for the chosen task
# and get the sim_and_plot parameters 
control_shell, runner_pars = task(arm_class, control_class)

# set up simulate and plot system
if args['--use_pygame'] is not None:
    from sim_and_plot_pygame import Runner
else:
    from sim_and_plot import Runner

runner = Runner(**runner_pars)

runner.run(arm=arm, control_shell=control_shell, video=args['--video'])
runner.show()
