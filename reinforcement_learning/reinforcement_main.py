# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:27:16 2017

@author: p000526832
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication
import argparse
import numpy as np
import sys
from chainerrl.optimizers import rmsprop_async
from chainerrl import links, policy, v_function
from chainerrl import misc
from chainerrl import experiments
from chainerrl.envs import ale
from chainerrl.agents import a3c
import chainerrl
import chainer.links as L
import chainer.functions as F
import chainer
from gym import spaces
from reinforcement_env import Reinforcement_Env, APPWINDOW
from builtins import *
from future import standard_library
standard_library.install_aliases()

# from dqn_phi import dqn_phi


NUM_EYES = 16
STATE_PAST_NUM = 4


class SState(object):
    def __init__(self, STATE_NUM, DIM):
        self.STATE_NUM = STATE_NUM
        self.DIM = DIM
        self.seq = np.zeros((STATE_NUM, DIM), dtype=np.float32)

    def push_s(self, state):
        self.seq[1:self.STATE_NUM] = self.seq[0:self.STATE_NUM - 1]
        self.seq[0][:NUM_EYES] = state[:NUM_EYES]

    def reset(self):
        self.seq = np.ones_like(self.seq) * 2

    def fill_s(self, state):
        for i in range(0, self.STATE_NUM):
            self.seq[i] = state


class Reinforcement_Env_Discrete(Reinforcement_Env):
    actions = np.array([[0.5, 1.0], [1.0, 1.0], [0.1, 0.5]])
    OBSDIM = 4

    def __init__(self, obs_agent):
        super().__init__(obs_agent)
        self.action_space_d = spaces.Discrete(self.actions.shape[0])
        self.observation_space_d = spaces.Discrete(self.OBSDIM)
        self.MyState = SState(self.OBSDIM, NUM_EYES)
        

    def _step(self, action):
        tmpState, tmpReward, tmpDone, tmpInfo = super().step(
            self.actions[action])
        self.MyState.push_s(tmpState)

        return self.MyState.seq.flatten(), tmpReward, tmpDone, tmpInfo

    def _reset(self):
        super().reset()
        self.MyState.reset()

        return self.MyState.seq.flatten()


class SimulationLoop(QThread):
    def __init__(self, agent, env):
        self.totaltime = 0
        QThread.__init__(self)
        self.agent = agent
        self.env = env

    def __del__(self):
        self.wait()

    def run(self):
        n_episodes = 10000
        max_episode_len = 10000

        if args.train:
            for i in range(1, n_episodes + 1):
                observation = self.env._reset()
                reward = 0
                R = 0
                for t in range(max_episode_len):
                    self.env.render()
                    act = self.agent.act_and_train(observation, reward)
                    observation, reward, done, info = self.env._step(act)
                    R += reward
                    if done:
                        break
                print(
                    "Episode {} finished after {} timesteps. reward {}".format(i, t+1, R))

                if i % 10 == 0:
                    print("episode:", i,
                          "R:", R,
                          "statistics:", self.agent.get_statistics())

                self.agent.stop_episode_and_train(observation, reward, done)

            self.agent.save('agent')
            print("Training completed.")

        if args.load_train:
            self.agent.load(args.load_train)

            for i in range(1, n_episodes + 1):
                observation = self.env.reset()
                reward = 0
                R = 0
                for t in range(max_episode_len):
                    self.env._render()

                    act = self.agent.act_and_train(observation, reward)
                    observation, reward, done, info = self.env._step(act)
                    R += reward

                    if done:
                        break
                print(
                    "Episode {} finished after {} timesteps. reward {}".format(i, t+1, R))

                if i % 10 == 0:
                    print("episode:", i,
                          "R:", R,
                          "statistics:", self.agent.get_statistics())

                self.agent.stop_episode_and_train(observation, reward, done)

            self.agent.save('agent')
            print("Training completed.")

        if args.load:
            self.agent.load(args.load)

            for i in range(100):
                observation = self.env._reset()
                reward = 0
                R = 0
                for t in range(max_episode_len):
                    self.env._render()
                    act = self.agent.act(observation)
                    observation, reward, done, info = self.env._step(act)
                    R += reward
                    if done:
                        print(
                            "Test Episode {} finished after {} timesteps".format(i, t+1))
                        break
                print('test episode:', i, 'R:', R)
                self.agent.stop_episode()
            print("Test completed. Now Waiting for closing the window.")


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super().__init__(
            l0=L.Linear(obs_size, 50),
            l1=L.Linear(50, 40),
            l2=L.Linear(40, 30),
            l3=L.Linear(30, n_actions)
        )

    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))

        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


class A3C_HEAD(chainer.Chain):
    def __init__(self, n_input_channels, n_out_put_channels):
        super().__init__(
            l0=L.Linear(n_input_channels, 50),
            l1=L.Linear(50, 40),
            l2=L.Linear(40, 30),
            l3=L.Linear(30, n_out_put_channels)
        )

    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))

        return self.l3(h)


class A3CFF(chainer.ChainList, a3c.A3CModel):

    def __init__(self, obs_num, n_actions):
        self.head = A3C_HEAD(obs_num, n_actions)
        self.pi = policy.FCSoftmaxPolicy(
            n_actions, n_actions)
        self.v = v_function.FCVFunction(n_actions)

        super().__init__(self.head, self.pi, self.v)

    def pi_and_v(self, state):
        out = self.head(state)
        return self.pi(out), self.v(out)


def CREATE_AGENT(env, agent_name):

    gamma = 0.9

    if agent_name == "DoubleDQN":

        q_func = QFunction(env.OBSDIM*(NUM_EYES), env.action_space_d.n)
        # q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
        #    env.OBSDIM*(NUM_EYES), env.action_space_d.n,
        #    n_hidden_layers=2, n_hidden_channels=50)

        # q_func.to_gpu(0)

        optimizer = chainer.optimizers.Adam(eps=1e-2)
        optimizer.setup(q_func)

        explorer = chainerrl.explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=env.action_space_d.sample)

        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

        agent = chainerrl.agents.DoubleDQN(
            q_func, optimizer, replay_buffer, gamma, explorer,
            replay_start_size=500, update_interval=1,
            target_update_interval=100)

        return agent

    if agent_name == "A3CFF":
        #        n_actions = ale.ALE(str(env.action_space_d.n)).number_of_actions
        #        model = A3CFF(n_actions)
        model = A3CFF(env.OBSDIM*(NUM_EYES), env.action_space_d.n)

        optimizer = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.9)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(40))

        agent = a3c.A3C(model, optimizer, t_max=4,
                        gamma=0.9, beta=1e-2, phi=dqn_phi)

        return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--load_train", "-lt", type=str, default=None)
    parser.add_argument("--load", "-l", type=str, default=None)
    parser.add_argument("--algotype", "-a", type=str, default="DoubleDQN")
    parser.add_argument("--obsagent", "-o", type=bool, default=0)
    parser.add_argument("--rom", "-r",  type=str)
    args = parser.parse_args()

    env = Reinforcement_Env_Discrete(args.obsagent)
    agent = CREATE_AGENT(env, args.algotype)

    print(args.obsagent)
    if args.train:
        # S = SimulationLoop(agent, env)
        # S.run()
        app = 0
        app = QApplication(sys.argv)

        w = APPWINDOW(args.obsagent, SimulationLoop(
            agent, env), title='Obstacle Acoidace')
        w.SetWorld(env)
        w.show()
        tmp = app.exec_()
        sys.exit(tmp)

    if args.load_train:
        S = SimulationLoop(agent, env)
        S.run()

    if args.load:
        app = 0
        app = QApplication(sys.argv)

        w = APPWINDOW(args.obsagent, SimulationLoop(
            agent, env), title='Obstacle Acoidace')
        w.SetWorld(env)
        w.show()

        tmp = app.exec_()
        sys.exit(tmp)
