""" Copyright (C) 2018 Travis DeWolf

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


Learning cart-pole with a policy network and value network.
The policy network learns the actions to take, and the value network learns
the expected reward from a given state, for use in calculating the advantage
function.

The value network is updated with a 2 norm loss * .5 without the sqrt on the
difference from the calculated expected cost. The value network is used to
calculated the advantage function at each point in time.

The policy network is updated by calculating the policy gradient and
advantage function.

Adapted from KVFrans:
github.com/kvfrans/openai-cartpole/blob/master/cartpole-policygradient.py
"""

import tensorflow as tf
import numpy as np
import random
import gym

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)

        # calculate the probability of the chosen action given the state
        good_probabilities = tf.reduce_sum(
            tf.multiply(probabilities, actions), reduction_indices=[1])
        # calculate the policy gradient
        eligibility = tf.multiply(tf.log(good_probabilities),
                                  tf.squeeze(advantages))
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])
        newvals = tf.placeholder("float", [None])
        w1 = tf.get_variable("w1", [4, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state, w1) + b1)
        w2 = tf.get_variable("w2", [10, 1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2

        # minimize the difference between predicted and actual
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    # unpack the policy network (generates control policy)
    (pl_calculated, pl_state, pl_actions,
        pl_advantages, pl_optimizer) = policy_grad
    # unpack the value network (estimates expected reward)
    (vl_calculated, vl_state, vl_newvals,
        vl_optimizer, vl_loss) = value_grad

    # set up the environment
    observation = env.reset()

    episode_reward = 0
    total_rewards = []
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    n_episodes = 0
    n_timesteps = 200
    for k in range(n_timesteps):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})

        # stochastically generate action using the policy output
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        episode_reward += reward

        # if the pole falls or time is up
        if done or k == n_timesteps - 1:
            for index, trans in enumerate(transitions):
                obs, action, reward = trans

                # calculate discounted monte-carlo return
                future_reward = 0
                future_transitions = len(transitions) - index
                decrease = 1
                for index2 in range(future_transitions):
                    future_reward += transitions[(index2) + index][2] * decrease
                    decrease = decrease * 0.97
                obs_vector = np.expand_dims(obs, axis=0)
                # compare the calculated expected reward to the average
                # expected reward, as estimated by the value network
                currentval = sess.run(vl_calculated,
                                      feed_dict={vl_state: obs_vector})[0][0]

                # advantage: how much better was this action than normal
                advantages.append(future_reward - currentval)

                # update the value function towards new return
                update_vals.append(future_reward)

            n_episodes += 1
            # reset variables for next episode in batch
            total_rewards.append(episode_reward)
            episode_reward = 0.0
            transitions = []

            if done:
                # if the pole fell, reset environment
                observation = env.reset()
            else:
                # if out of time, close environment
                env.close()

    print('total_rewards: ', total_rewards)

    # update value function
    sess.run(vl_optimizer,
             feed_dict={vl_state: states,
                        vl_newvals: update_vals})

    # update control policy
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states,
                                      pl_advantages: advantages_vector,
                                      pl_actions: actions})

    return total_rewards, n_episodes

# generate the networks
policy_grad = policy_gradient()
value_grad = value_gradient()

# run the training from scratch 10 times, record results
for ii in range(10):

    env = gym.make('CartPole-v0')
    env = gym.wrappers.Monitor(
        env=env,
        directory='cartpole-hill/',
        force=True,
        video_callable=False)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    max_rewards = []
    total_episodes = []
    # each batch is 200 time steps worth of episodes
    n_training_batches = 300
    import time
    times = []
    for i in range(n_training_batches):
        start_time = time.time()
        if i % 100 == 0:
            print(i)
        reward, n_episodes = run_episode(env, policy_grad, value_grad, sess)
        max_rewards.append(np.max(reward))
        total_episodes.append(n_episodes)
        times.append(time.time() - start_time)
    print('average time: %.3f' % (np.sum(times) / n_training_batches))

    np.savez_compressed('data/policy_gradient_%i' % ii,
        max_rewards=max_rewards, total_episodes=total_episodes)

    sess.close()
