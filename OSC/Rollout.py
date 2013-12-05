#Written by Travis DeWolf (November 2013)
import numpy as np

"""from Controllers.TwoLinkArm.control import Control
from Arms.TwoLinkArm.arm_python import Arm

#dt=.01
#dmps = DMPs_discrete(dmps=2, bfs=2000, dt=dt)
#controller = Control()
#arm = Arm(dt=.00001)

'''timesteps = int(dmps.run_time/dmps.dt)
y_sys = np.zeros((timesteps,2))
y_dmp = np.zeros((timesteps,2))
kp = 1000; kv = np.sqrt(kp)
for tt in range(timesteps):
    x = arm.position(ee_only=True)
    y, dy, ddy = dmps.step()#state_fb=x)
  
    JEE = controller.gen_jacEE(arm) 
    dx = np.dot(JEE, arm.dq) 

    #u = ddy + kp * (y - x) + kv * (dy - dx)
    #u = np.ones(dmps.dmps)*.1
    u = ddy
    u = controller.control_osc(arm, u_x=u)

    y_sys[tt] = arm.apply_torque(u=u)
    y_dmp[tt] = y'''

import Controllers.Trajectories.read_trajectory as rt
box = [-.3, .3, .25, .45]
trajectory = rt.read_file('Controllers/Trajectories/ca0.dat', '', box=box)
#trajectory = np.sin(np.linspace(0,1,100)*10)
#trajectory = np.array([trajectory, trajectory]).T
#dmps.imitate_path(y_des=trajectory.T)

import Controllers.Trajectories.DMP1 as DMP1
DMP1 = DMP1.DMPs_discrete(DMP_NUM = 2, BF_NUM = 200, 
                        total_time=len(trajectory),
                        y0=np.zeros(2), goal=np.ones(2))
DMP1.imitate_paths(dt=1.0, end_time=len(trajectory), 
                  y_des=trajectory.T)
dt = .02
ytrack,_,u_des = DMP1.open_rollout(dt=dt, end_time=int(1./dt)* \
                                                   len(trajectory))
import Controllers.Trajectories.DMP as DMP
dmp = DMP.DMPs_discrete(dmps=2, bfs=200, dt=.01)
p = dmp.imitate_path(y_des=trajectory.T)
y,dy,ddy = dmp.rollout()

import matplotlib.pyplot as plt
plt.plot(y[:,0], y[:,1])
plt.plot(p[0], p[1])
#plt.plot(ytrack[:,0], ytrack[:,1])
#plt.plot(trajectory[:,0], trajectory[:,1])
#plt.plot(ddy)
#plt.plot(u_des)
plt.grid()
plt.legend(['new DMP', 'old DMP', 'desired'])
plt.show()

'''import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
#plt.plot(y_sys)
a = plt.plot(y_dmp[:,0], y_dmp[:,1], 'b', lw=2)
a = plt.plot(y_sys[:,0], y_sys[:,1], 'g', lw=2)
b = plt.plot(trajectory[:,0], trajectory[:,1], 'r')
#plt.legend([a[0], b[0]], ['DMP', 'desired'], loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()'''"""

import Controllers.Trajectories.read_trajectory as rt
box = [-.3, .3, .25, .45]
trajectory = rt.read_file('Controllers/Trajectories/ca0.dat', '', box=box)

import Controllers.Trajectories.DMP as DMP

import time
s_time = time.time()
dmp = DMP.DMPs_discrete(dmps=2, bfs=100)
#path = np.sin(np.linspace(0,1,100)*10)
#path = np.array([path, path])
path = dmp.imitate_path(trajectory.T)
timesteps = 50000
y_track = np.zeros((timesteps,2))
for i in range(timesteps):
    y_track[i], dy, ddy = dmp.step(tau=.005)
#y_track,dy_track,ddy_track = dmp.rollout(tau=.005)
print time.time() - s_time

import matplotlib.pyplot as plt
plt.plot(y_track[:,0], y_track[:,1])
plt.plot(path[0], path[1], 'r')
'''plt.subplot(321)
plt.ylabel('y_track')
plt.plot(y_track[:,0])
plt.plot(y1_track[:,0])
plt.subplot(322)
plt.plot(y_track[:,1])
plt.plot(y1_track[:,1])
plt.subplot(323)
plt.ylabel('dy_track')
plt.plot(dy_track[:,0]/max(abs(dy_track[:,0])))
plt.plot(dy1_track[:,0]/max(abs(dy1_track[:,0])))
plt.subplot(324)
plt.plot(dy_track[:,1]*.02)
plt.plot(dy1_track[:,1])
plt.subplot(325)
plt.ylabel('ddy_track')
plt.plot(ddy_track[:,0]*.02)
plt.plot(ddy1_track[:,0])
plt.subplot(326)
plt.plot(ddy_track[:,1])
plt.plot(ddy1_track[:,1])'''
plt.tight_layout()
plt.show()
