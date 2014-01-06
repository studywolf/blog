'''
Copyright (C) 2013 Terry Stewart & Travis DeWolf

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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Runner:
    def __init__(self, title='', dt=1e-5, 
                       control_steps=10, display_steps=200, 
                       t_target=0.5, max_tau=100, 
                       seed=1, box=[-1,1,-1,1],
                       control_type=None, 
                       trajectory=None):
        self.dt = dt
        self.control_steps = control_steps
        self.display_steps = display_steps
        self.max_tau = max_tau
        self.target_steps = int(t_target/float(dt*display_steps))
        self.trajectory = trajectory

        self.box = box 
        self.control_type = control_type 
        self.mouse_control_active = False
        self.title = title

        self.sim_step = 0
        
    def run(self, arm, control, video=None, video_time=None):
        self.arm = arm
        self.control = control
        
        fig = plt.figure(figsize=(5.1,5.1), dpi=None)
        fig.suptitle(self.title); 
        # set the padding of the subplot explicitly
        fig.subplotpars.left=.1; fig.subplotpars.right=.9
        fig.subplotpars.bottom=.1; fig.subplotpars.top=.9

        ax = fig.add_subplot(1, 1, 1, 
                xlim=(self.box[0], self.box[1]), 
                ylim=(self.box[2], self.box[3]))
        ax.xaxis.grid(); ax.yaxis.grid()
        # make it a square plot
        ax.set_aspect(1) 

        # set up plot elements
        self.arm_line, = ax.plot([], [], 'o-', mew=4, color='b', lw=5)
        self.target_line, = ax.plot([], [], 'r-x', mew=4)
        self.trail, = ax.plot([], [], color='#888888', lw=3)
        self.info = ax.text(self.box[0]+abs(.1*self.box[0]), \
                            self.box[3]-abs(.1*self.box[3]), \
                            '', va='top')
        self.trail_data = np.ones((self.target_steps*200, 2), \
                                    dtype='float') * np.NAN
    
        if self.trajectory is not None:
            ax.plot(self.trajectory[:,0], self.trajectory[:,1], alpha=.3)

        # connect up mouse event if correct for control type
        if self.control_type[:3] == 'osc': 
            # get pixel width of fig (-.2 for the padding)
            self.fig_width = (fig.get_figwidth() - .2 \
                                * fig.get_figwidth()) * fig.get_dpi()
            def move_target(event): 
                self.mouse_control_active = True
                # get mouse position and scale appropriately to convert to (x,y) 
                target = ((np.array([event.x, event.y]) - .5 * fig.get_dpi()) /\
                                self.fig_width) * \
                                (self.box[1] - self.box[0]) + self.box[0]

                # set target for the controller
                self.target = self.control.set_target_from_mouse(target)

            # hook up function to mouse event
            fig.canvas.mpl_connect('motion_notify_event', move_target)

        if video_time is None:
            frames = 50
        else:
            frames = int(video_time/(self.dt*self.display_steps))

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                   init_func=self.anim_init, frames=150, interval=0, blit=True)
        
        if video is not None:
            anim.save(video, fps=1.0/(self.dt*self.display_steps), dpi=200)
        
        self.anim = anim
        
    def make_info_text(self):
        text = []
        text.append('t = %1.4g'%(self.sim_step*self.dt))
        u_text = ' '.join('%4.3f,'%F for F in self.control.u)
        text.append('u = ['+u_text+']')
                
        return '\n'.join(text)    

    def anim_init(self):
        self.info.set_text('')
        self.arm_line.set_data([], [])
        self.target_line.set_data([], [])
        self.trail.set_data([], [])
        return self.arm_line, self.target_line, self.info, self.trail

    def anim_animate(self, i):

        if self.control_type in ('trajectory', 'dmp'):
            self.target = self.control.target
        elif not self.mouse_control_active:
            # update target
            if self.sim_step % (self.target_steps*self.display_steps) == 0:
                self.target = self.control.gen_target(self.arm)
                # update target plot
                print 'mouse_target: ', self.target
       
        # before drawing
        for j in range(self.display_steps):            
            # update control signal
            if self.sim_step % self.control_steps == 0 or \
                'tau' not in locals():
                tau = self.control.control(self.arm)
            # apply control signal and simulate
            self.arm.apply_torque(u=tau, dt=self.dt)
    
            self.sim_step +=1
        
        # update hand trail
        self.trail_data[:-1] = self.trail_data[1:]
        if self.control.pen_down:
            self.trail_data[-1] = self.arm.x
        else: self.trail_data[-1] = [None, None]

        # update figure
        self.arm_line.set_data(*self.arm.position())
        if self.target is not None: self.target_line.set_data(self.target)
        self.info.set_text(self.make_info_text())
        self.trail.set_data(self.trail_data[:,0], self.trail_data[:,1])
        return self.arm_line, self.target_line, self.info, self.trail

    def show(self):
        try:
            plt.show()
        except AttributeError:
            pass
