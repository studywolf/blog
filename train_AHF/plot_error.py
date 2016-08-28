'''
Copyright (C) 2016 Travis DeWolf

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
import matplotlib.pyplot as plt
import glob
import sys
import seaborn


def gen_data_plot(folder="weights", index=None, show_plot=True,
                  save_plot=None, save_paths=False, verbose=True):

    files = sorted(glob.glob('%s/rnn*' % folder))
    files = files[:index] if index is not None else files

    # plot the values over time
    vals = []

    for ii, name in enumerate(files):
        if verbose:
            print(name)
        name = name.split('err')[1]
        name = name.split('.npz')[0]
        vals.append(float(name))

    vals = np.array(vals)

    plt.figure(figsize=(10, 3))
    ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    ax.loglog(vals)
    ax.loglog(range(len(vals)), np.ones(len(vals)) * min(vals), 'r--')
    ax.loglog(range(len(vals)), np.ones(len(vals)) * min(vals), 'r--')
    plt.xlim([0, len(files)])
    plt.ylim([10**-5, 10])
    plt.title('AHF training error')
    plt.xlabel('Training iterations')
    plt.ylabel('Error')
    plt.yscale('log')

    # load in the weights and see how well they control the arm
    dt = 1e-2
    sig_len = 40

    # HACK: append system path to have access to the arm code
    # NOTE: Change this path to wherever your plant model is kept!
    sys.path.append("../../../studywolf_control/studywolf_control/")
    # from arms.two_link.arm_python import Arm as Arm
    from arms.three_link.arm import Arm as Arm
    if verbose:
        print('Plant is: %s' % str(Arm))
    arm = Arm(dt=dt)

    from hessianfree import RNNet
    from hessianfree.nonlinearities import (Tanh, Linear)
    from train_hf_3link import PlantArm, gen_targets

    rec_coeff = [1, 1]
    rec_type = "sparse"
    eps = 1e-6

    num_states = arm.DOF * 2
    targets = gen_targets(arm, sig_len=sig_len)
    init_state = np.zeros((len(targets), num_states), dtype=np.float32)
    init_state[:, :arm.DOF] = arm.init_q # set up the initial joint angles
    plant = PlantArm(arm, targets=targets,
                     init_state=init_state, eps=eps)

    index = -1 if index is None else index
    W = np.load(files[index])['arr_0']

    # make sure this network is the same as the one you trained!
    net_size = 96
    if '32' in folder:
        net_size = 32
    rnn = RNNet(shape=[num_states * 2,
                       net_size,
                       net_size,
                       num_states,
                       num_states],
                layers=[Linear(), Tanh(), Tanh(), Linear(), plant],
                debug=False,
                rec_layers=[1, 2],
                conns={0: [1, 2], 1: [2], 2: [3], 3: [4]},
                W_rec_params={"coeff": rec_coeff, "init_type": rec_type},
                load_weights=W,
                use_GPU=False)

    rnn.forward(plant, rnn.W)
    states = np.asarray(plant.get_vecs()[0][:, :, num_states:])
    targets = np.asarray(plant.get_vecs()[1])

    def kin(q):
        x = np.sum([arm.L[ii] * np.cos(np.sum(q[:, :ii+1], axis=1))
                    for ii in range(arm.DOF)], axis=0)
        y = np.sum([arm.L[ii] * np.sin(np.sum(q[:, :ii+1], axis=1))
                    for ii in range(arm.DOF)], axis=0)
        return x,y

    ax = plt.subplot2grid((1, 3), (0, 2))
    # plot start point
    initx, inity = kin(init_state)
    ax.plot(initx, inity, 'x', mew=10)
    for jj in range(0, len(targets)):
        # plot target
        targetx, targety = kin(targets[jj])
        ax.plot(targetx, targety, 'rx', mew=1)
        # plat path
        pathx, pathy = kin(states[jj, :, :])
        path = np.hstack([pathx[:, None], pathy[:, None]])
        if save_paths is True:
            np.savez_compressed('end-effector position%.3i.npz' % int(jj/8),
                                array1=path)
        ax.plot(path[:, 0], path[:, 1])

    plt.tight_layout()
    # plt.xlim([-.1, .1])
    # plt.ylim([.25, .45])
    plt.title('Hand trajectory')
    plt.xlabel('x')
    plt.ylabel('y')

    if save_plot is not None:
        plt.savefig(save_plot)
    if show_plot is True:
        plt.show()
    plt.close()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        folder = "weights"
    else:
        folder = sys.argv[1]
    if len(sys.argv) < 3:
        index = None
    else:
        index = int(sys.argv[2])

    gen_data_plot(folder=folder, index=index)
