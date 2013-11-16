#Written by Travis DeWolf (November 2013)
import numpy as np

class CanonicalSystem():
    """Implementation of the canonical dynamical system
    as described in Dr. Stefan Schaal's (2002) paper"""

    def __init__(self, ax=1.):
        """Default values from Schaal (2012)
        
        ae float: coefficient on phase activation
        """
        self.ax = ax
        self.x = 1.0

    def discrete_rollout(self, dt, run_time):
        """Generate x for discrete open loop movements.
        Decaying from 1 to 0 according to dx = -ax*x.

        dt float: timestep
        run_time float: how long to run the CS
        """
        timesteps = int(run_time / dt)
        self.x_track = np.zeros(timesteps)
        
        self.x = 1.0
        for t in range(timesteps):
            self.x_track[t] = self.x 
            self.discrete_step(dt)

        return self.x_track

    def discrete_step(self, dt):
        """Generate a single step of x for discrete
        closed loop movements. Decaying from 1 to 0 
        according to dx = -ax*x.

        dt float: timestep
        """
        self.x += (-self.ax * self.x) * dt
        return self.x
        

#==============================
# Test code
#==============================
if __name__ == "__main__":
    
    cs = CanonicalSystem()
    x_track = cs.discrete_rollout(dt=.001, run_time=5.0)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3))
    plt.plot(x_track, lw=2)
    plt.title('Canonical system')
    plt.xlabel('time (s)')
    plt.ylabel('x')
    plt.tight_layout()
    plt.show()
