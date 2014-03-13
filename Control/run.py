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
    --video=VIDPARS  (title, num_frames) [default:('vid.mp4',100)]

'''

from Arms.one_link.arm import Arm1Link as Arm1
from Arms.two_link.arm import Arm2Link as Arm2
from Arms.two_link.arm_python import Arm2Link as Arm2Python
from Arms.three_link.arm import Arm3Link as Arm3

import Controllers.dmp as DMP
import Controllers.gc as GC
import Controllers.osc as OSC
import Controllers.trajectory as Trajectory

import Tasks.follow_mouse as follow_mouse
import Tasks.random_movements as random_movements
import Tasks.reach as reach
import Tasks.write_numbers as write_numbers
import Tasks.write_words as write_words
import Tasks.walk as walk

from sim_and_plot import Runner

from docopt import docopt
import numpy as np

args = docopt(__doc__)

# get and instantiate the chosen arm
arm_class = {'arm1':Arm1,
             'arm2':Arm2,
             'arm2_python':Arm2Python,
             'arm3':Arm3}[args['ARM']]
arm = arm_class(options=args['--arm_options'])

# get the chosen controller class
control_class = {'dmp':DMP,
           'gc':GC.Control,
           'osc':OSC.Control,
           'trajectory':Trajectory}[args['CONTROL']]

# get the chosen task class
task = {'follow':follow_mouse.Task,
        'random':random_movements.Task,
        'reach':reach.Task,
        'write_numbers':write_numbers.Task,
        'write_words':write_words.Task,
        'walk':walk.Task}[args['TASK']]

# instantiate the controller for the chosen task
# and get the sim_and_plot parameters 
controller, runner_pars = task(arm_class, control_class)

# set up mouse control
runner = Runner(**runner_pars)

runner.run(arm=arm, control=controller, video=args['--video'])
runner.show()
