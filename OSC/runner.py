import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Runner:
    def __init__(self, title, dt=1e-5, control_steps=10, display_steps=200, 
                       t_target=0.5, max_tau=100, seed=1, xylim=1,
                       show_error=True):
        self.dt = dt
        self.control_steps = control_steps
        self.display_steps = display_steps
        self.target_steps = int(t_target/float(dt*display_steps))
        self.max_tau = max_tau

        self.show_error = show_error
        self.error = None
        self.errors = []
        self.title = title
        self.xylim = xylim

        self.sim_step = 0
        
    def run(self, arm, control, video=None, video_time=None):
        self.arm = arm
        self.control = control
        
        fig = plt.figure(figsize=(5.1,5.1))
        fig.suptitle(self.title)
        ax = fig.add_subplot(1, 1, 1, 
                xlim=(-self.xylim, self.xylim), ylim=(-self.xylim, self.xylim))
        ax.xaxis.grid(); ax.yaxis.grid()
        ax.set_aspect(1) # make it a square plot
        self.arm_line, = ax.plot([], [], 'o-', mew=4, color='b', lw=5)
        self.target_line, = ax.plot([], [], 'r-x', mew=4)
        self.trail, = ax.plot([], [], color='#888888')
        self.info = ax.text(-self.xylim+.1, self.xylim-.1, '', va='top')
        self.trail_data = np.ones((self.target_steps, 2), dtype='float') * arm.x

        if video_time is None:
            frames = 50
        else:
            frames = int(video_time/(self.dt*self.display_steps))

        anim = animation.FuncAnimation(fig, self.anim_animate, 
                   init_func=self.anim_init, frames=None, interval=0, blit=True)
        
        if video is not None:
            anim.save(video, fps=1.0/(self.dt*self.display_steps))
        
        self.anim = anim
        
    def make_info_text(self):
        text = []
        text.append('t = %1.4g'%(self.sim_step*self.dt))
        u_text = ' '.join('%4.3f,'%F for F in self.control.u)
        text.append('u = ['+u_text+']')
        if self.show_error:
            if self.error is not None:
                text.append('error = %1.3f'%self.error)
                
        return '\n'.join(text)    

    def anim_init(self):
        self.info.set_text('')
        self.arm_line.set_data([], [])
        self.target_line.set_data([], [])
        self.trail.set_data([], [])
        return self.arm_line, self.target_line, self.info, self.trail

    def anim_animate(self, i):
        # update target
        if self.sim_step % (self.target_steps*self.display_steps) == 0:
            
            self.target = self.control.gen_target(self.arm)
            # update target plot
            print self.target
       
        # before drawing
        for j in range(self.display_steps):            
            # update control signal
            if self.sim_step % self.control_steps == 0:
                tau = self.control.control(self.arm)
            # apply control signal and simulate
            self.arm.apply_torque(u=tau, dt=self.dt)
    
            self.sim_step +=1
        
        # update hand trail   
        self.trail_data[:-1] = self.trail_data[1:]
        self.trail_data[-1] = self.arm.x
        # update figure
        self.arm_line.set_data(*self.arm.position())
        self.target_line.set_data(self.target)
        self.info.set_text(self.make_info_text())
        self.trail.set_data(self.trail_data[:,0], self.trail_data[:,1])
        return self.arm_line, self.target_line, self.info, self.trail

    def show(self):
        try:
            plt.show()
        except AttributeError:
            pass
