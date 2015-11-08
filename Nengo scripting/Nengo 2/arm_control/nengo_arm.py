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
'''

import nengo
import numpy as np

from Arm import Arm2Link

run_in_GUI = True 

# set the initial position of the arm
arm = Arm2Link()
arm.reset(q=[np.pi/5.5, np.pi/1.7], dq=[0, 0])

"""Generate the Nengo model that will control the arm."""
model = nengo.Network()

with model: 
    
    # create input nodes
    def arm_func(t, x):
        u = x[:2]
        arm.apply_torque(u) 
        data = np.hstack([arm.q0, arm.q1, arm.dq0, arm.dq1, arm.x]) # data returned from node to model

        # visualization code -----------------------------------------------------
        scale = 30
        len0 = arm.l1 * scale
        len1 = arm.l2 * scale
        
        angles = data[:3]
        angle_offset = np.pi/2
        x1 = 50
        y1 = 100
        x2 = x1 + len0 * np.sin(angle_offset-angles[0])
        y2 = y1 - len0 * np.cos(angle_offset-angles[0])
        x3 = x2 + len1 * np.sin(angle_offset-angles[0] - angles[1])
        y3 = y2 - len1 * np.cos(angle_offset-angles[0] - angles[1])

        arm_func._nengo_html_ = '''
        <svg width="100%" height="100%" viewbox="0 0 100 100">
            <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="stroke:black"/>
            <line x1="{x2}" y1="{y2}" x2="{x3}" y2="{y3}" style="stroke:black"/>
            <circle cx="{x3}" cy="{y3}" r="1.5" stroke="black" stroke-width="1" fill="black" />
        </svg>
        '''.format(**locals())
        # end of visualization code ---------------------------------------------

        return data
    arm_node = nengo.Node(output=arm_func, size_in=2)

    # specify torque input to arm
    input_node = nengo.Node(output=[1, .1])

    # to send a target to an ensemble which then connections to the arm
    ens = nengo.Ensemble(n_neurons=500, dimensions=2, radius=np.sqrt(20))
    nengo.Connection(input_node, ens[:2]) # to send target info to ensemble

    # connect ens to arm
    nengo.Connection(ens, arm_node)#, function=some_function) 
    # --------------------------------------------------------

if run_in_GUI:
    # to run in GUI, comment out next 4 lines for running without GUI
    import nengo_gui
    nengo_gui.GUI(model=model, filename=__file__, locals=locals(), 
                  interactive=False, allow_file_change=False).start()
    import sys
    sys.exit()
else:  
    # to run in command line
    with model:    
        probe_input = nengo.Probe(input_node)
        probe_arm = nengo.Probe(arm_node[arm.DOF*2])
        
    print 'building model...'
    sim = nengo.Simulator(model, dt=.001)
    print 'build complete.'

    sim.run(10)

    t = sim.trange()
    x = sim.data[probe_arm]
    y = sim.data[probe_arm]

    # plot collected data
    import matplotlib.pyplot as plt

    plt.subplot(311)
    plt.plot(t, x)
    plt.xlabel('time')
    plt.ylabel('probe_arm0')

    plt.subplot(312)
    plt.plot(t, y)
    plt.xlabel('time')
    plt.ylabel('probe_arm1')

    plt.subplot(313)
    plt.plot(x, y)
    plt.xlabel('probe_arm0')
    plt.ylabel('probe_arm1')

    plt.tight_layout()
    plt.show()

