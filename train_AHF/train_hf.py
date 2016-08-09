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

from hessianfree import RNNet
from hessianfree.optimizers import HessianFree 
from hessianfree.nonlinearities import (Tanh, Linear, Plant)

class PlantFD(Plant):
    """ An extension of Plant that implements Finite Differences 
    for calculating d_input and d_state """
    def __init__(self, num_states, targets, init_state, eps):
        super(Plant, self).__init__(stateful=True)

        self.eps = eps

        self.targets = targets
        self.init_state = init_state

        self.shape = [targets.shape[0],
                      targets.shape[1],
                      num_states]

        # derivative of output with respect to state (constant, so just
        # compute it once here)
        self.d_output = np.resize(np.eye(self.shape[-1]),
                                  (targets.shape[0], self.shape[-1],
                                   self.shape[-1], 1))

        self.reset()

    def activation(self, x, update=True):
        raise NotImplementedError

    def __call__(self, x):
        # feed in the final target state and current system state as input
        inputs = np.concatenate([np.nan_to_num(self.targets[:, -1]), 
                                 self.state], axis=1)
        self.inputs = np.concatenate((self.inputs, inputs[:, None, :]), axis=1)
        return inputs

    # TODO: give option between forward / backward / central differencing
    def d_activation(self, x, a):
        self.d_act_count += 1
        assert self.act_count == self.d_act_count

        state_backup = np.copy(self.state)
        # calculate ds0/dx0 with finite differences
        d_input_FD = np.zeros((x.shape[0], x.shape[1], self.state.shape[1]))
        for ii in range(x.shape[1]):
            # calculate state adding eps to x[ii]
            self.reset_plant(self.prev_state)
            inc_x = x.copy()
            inc_x[:, ii] += self.eps
            self.activation(inc_x, update=False)
            state_inc = self.state.copy()
            # calculate state subtracting eps from x[ii]
            self.reset_plant(self.prev_state)
            dec_x = x.copy()
            dec_x[:, ii] -= self.eps
            self.activation(dec_x, update=False)
            state_dec = self.state.copy()

            d_input_FD[:, :, ii] = (state_inc - state_dec) / (2 * self.eps)
        d_input_FD = d_input_FD[..., None]

        # calculate ds1/ds0
        d_state_FD = np.zeros((x.shape[0], self.state.shape[1], self.state.shape[1]))
        for ii in range(self.state.shape[1]):
            # calculate state adding eps to self.state[ii]
            state = np.copy(self.prev_state)
            state[:, ii] += self.eps
            self.reset_plant(state)
            self.activation(x, update=False)
            state_inc = self.state.copy()
            # calculate state subtracting eps from self.state[ii]
            state = np.copy(self.prev_state)
            state[:, ii] -= self.eps
            self.reset_plant(state)
            self.activation(x, update=False)
            state_dec = self.state.copy()

            d_state_FD[:, :, ii] = (state_inc - state_dec) / (2 * self.eps)
        d_state_FD = d_state_FD[..., None]
        self.reset_plant(state_backup)

        return np.concatenate((d_input_FD, d_state_FD, self.d_output), axis=-1)

    def get_vecs(self):
        return (self.inputs, self.targets)

    def reset(self, init_state=None):
        self.act_count = 0
        self.d_act_count = 0
        self.reset_plant(self.init_state.copy() if init_state is None else
                          init_state.copy())
        # * 2 because we provide both the current system state and targets
        self.inputs = np.zeros((self.shape[0], 0, self.shape[-1] * 2),
                               dtype=np.float32)

    def reset_plant(self, state):
        raise NotImplementedError


class PlantArm(PlantFD):
    """ Runs a given arm model as the plant """
    def __init__(self, arm, targets, **kwargs):
        # create an arm for each target / run
        self.arm = arm
        PlantFD.__init__(self, num_states=self.arm.DOF*2, 
                        targets=targets, **kwargs)

    def activation(self, x, update=True):

        # use all the network output is the control signal
        u = np.array([np.sum(x[:, ii::self.arm.DOF], axis=1)
                for ii in range(self.arm.DOF)]).T

        state = []
        for ii in range(x.shape[0]):
            self.arm.reset(q=self.state[ii, :self.arm.DOF], 
                           dq=self.state[ii, self.arm.DOF:])
            self.arm.apply_torque(u[ii])
            state.append(np.hstack([self.arm.q, self.arm.dq]))
        state = np.asarray(state)

        if update is True:
            self.act_count += 1
            self.prev_state = self.state.copy()
        self.state = self.squashing(state)

        if np.isnan(np.sum(self.state)):
            print(self.state)
            raise Exception
        return self.state[:x.shape[0]]
        # NOTE: generally x will be the same shape as state, this just
        # handles the case where we're passed a single item instead
        # of batch)

    def reset_plant(self, state):
        # set all the arm states to state
        self.state = np.copy(state)

    def squashing(self, x):
        index_below = np.where(x < -2*np.pi)
        x[index_below] = np.tanh(x[index_below]+2*np.pi) - 2*np.pi
        index_above = np.where(x > 2*np.pi)
        x[index_above] = np.tanh(x[index_above]-2*np.pi) + 2*np.pi
        return x

def gen_targets(arm, n_targets=8, sig_len=100):
    """ Generate target angles corresponding to target 
    (x,y) coordinates around a circle """
    import scipy.optimize

    x_bias = 0
    if arm.DOF == 2:
        y_bias = .35
        dist = .075
    elif arm.DOF == 3:
        y_bias = .5
        dist = .2

    # set up the reaching trajectories around circle
    targets_x = [dist * np.cos(theta) + x_bias \
                    for theta in np.linspace(0, np.pi*2, 65)][:-1]
    targets_y = [dist * np.sin(theta) + y_bias \
                    for theta in np.linspace(0, np.pi*2, 65)][:-1]

    joint_targets = []
    for ii in range(len(targets_x)):
        joint_targets.append(arm.inv_kinematics(xy=(targets_x[ii],
                                                    targets_y[ii])))
    targs = np.asarray(joint_targets)
   
    for ii in range(targs.shape[1]-1):
        targets = np.concatenate(
            (np.outer(targs[:, ii], np.ones(sig_len))[:, :, None],
             np.outer(targs[:, ii+1], np.ones(sig_len))[:, :, None]), axis=-1)
    targets = np.concatenate((targets, np.zeros(targets.shape)), axis=-1)
    # only want to penalize the system for not being at the 
    # target at the final state, set everything before to np.nan
    targets[:, :-1] = np.nan 

    return targets

def test_plant():
    """Example of a network using a dynamic plant as the output layer."""

    eps = 1e-6 # value to use for finite differences computations
    dt = 1e-2 # size of time step 
    sig_len = 40 # how many time steps to train over
    batch_size = 32 # how many updates to perform with static input
    num_batches = 20000 # how many batches to run total

    import sys
    # NOTE: Change to wherever you keep your arm models
    sys.path.append("../../../studywolf_control/studywolf_control/")
    from arms.two_link.arm_python import Arm as Arm
    print('Plant is: %s' % str(Arm))
    arm = Arm(dt=dt, init_q=[0.736134824578, 1.85227640003])

    num_states = arm.DOF * 2 # are states are [positions, velocities]
    targets = gen_targets(arm=arm, sig_len=sig_len) # target joint angles
    init_state = np.zeros((len(targets), num_states)) # initial velocity = 0
    init_state[:, :arm.DOF] = arm.init_q # set up the initial joint angles 
    plant = PlantArm(arm=arm, targets=targets, 
                        init_state=init_state, eps=eps)

    # open up weights folder and checked for saved weights
    import glob
    files = sorted(glob.glob('weights/rnn*'))
    if len(files) > 0:
        # if weights found, load them up and keep going from last trial
        W = np.load(files[-1])['arr_0'] 
        print('loading from %s' % files[-1])
        last_trial = int(files[-1].split('weights/rnn_weights-trial')[1].split('-err')[0])
        print('last_trial: %i' % last_trial)
    else:
        # if no weights found, start fresh with new random seed
        W = None
        last_trial = -1
        seed = np.random.randint(100000000)
        print('seed : %i' % seed)
        np.random.seed(seed) 

    # specify the network structure and loss functions
    from hessianfree.loss_funcs import SquaredError, SparseL2
    rnn = RNNet(
        # specify the number of nodes in each layer
        shape=[num_states * 2, 32, 32, num_states, num_states], 
        # specify the function of the nodes in each layer
        layers=[Linear(), Tanh(), Tanh(), Linear(), plant],
        # specify the layers that have recurrent connections
        rec_layers=[1,2],
        # specify the connections between layers
        conns={0:[1, 2], 1:[2], 2:[3], 3:[4]},
        # specify the loss function
        loss_type=[
            # squared error between plant output and targets
            SquaredError()],
        load_weights=W,
        use_GPU=False)

    # set up masking so that weights between network output
    # and the plant aren't modified in learning, always = 1
    offset, W_end, b_end = rnn.offsets[(3,4)]
    rnn.mask = np.zeros(rnn.W.shape, dtype=bool)
    rnn.mask[offset:b_end] = True
    rnn.W[offset:W_end] = np.eye(4).flatten()

    for ii in range(last_trial+1, num_batches):
        print('=============================================')
        print('training batch %i' % ii)
        err = rnn.run_epochs(plant, None, max_epochs=batch_size,
                        optimizer=HessianFree(CG_iter=96, init_damping=100))
        # save the weights to file, track trial and error 
        err = rnn.best_error
        name = 'weights/rnn_weights-trial%04i-err%.5f'%(ii, err) 
        np.savez_compressed(name, rnn.W)
        print('=============================================')
        print('network: %s' % name)
        print('final error: %f' % err)
        print('=============================================')

    return rnn.best_error

if __name__ == '__main__':
    test_plant()
