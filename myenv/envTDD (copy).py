import sys

import gym
import numpy as np
import gym.spaces

import math
import random

from hetnet import HetNet

class MyEnvTDD(gym.Env):

    #MAX_STEPS = 100000
    #MAX_STEPS = 4000
    #MAX_STEPS = 1999
    #MAX_STEPS = 999
    MAX_STEPS = 1

    MAX_CAPACITY = 100
    
    MAX_RATE = 100

    def __init__(self, sim, BStype, id_x, observation_len, s_seed):
        super().__init__()
        #random.seed(0)

        self.sim = sim

        self.bs = self.sim.add_bs(BStype, id_x)

        print(str(self.bs.get_name()) + ": init")

        observation_space_tmp = []
        for i in range(observation_len):
            observation_space_tmp.append(self.MAX_RATE)
        # action_space, observation_space, reward_range を設定する
        #self.action_space = gym.spaces.Discrete(self.NUM_RRH * 2)   # ON/ OF
        self.action_space = gym.spaces.Discrete(7)   # ON/ OF
        self.observation_space = gym.spaces.MultiDiscrete(observation_space_tmp) # M1, ..., MR, D1, ...., DR
        self.reward_range = [0.0, self.MAX_CAPACITY]

        print(str(self.bs.get_name()) + ": self.action_space"+str(self.action_space))
        print(str(self.bs.get_name()) + ": self.observation_space"+str(self.observation_space))

        random.seed(s_seed)
        np.random.seed(seed=s_seed)

        print(str(self.bs.get_name()) + ": get_ue_num")
        print(str(self.bs.get_name()) + ": " + str(self.sim.get_ue_num()))

        random.seed(0)
        self.observation = np.ones(self.sim.get_ue_num(), dtype=int)


        self._reset()


    def _reset(self):
        print(str(self.bs.get_name()) + ": reset")

        self.steps = 0        

        return self._observe()

    def _step(self, action):
        print(str(self.bs.get_name()) + ": step")

        print(str(self.bs.get_name()) + ": step, steps: "+str(self.steps)+ ", "+str(self.steps))
        print(str(self.bs.get_name()) + ": step, action: "+str(self.steps)+ ", "+str(action))
 
        self.output = action % 7

        print(str(self.bs.get_name()) + ": step, self.output: "+str(self.steps)+ ", "+str(self.output))

        #0 stay, 1 on, 2 off


        self.bs.set_config(self.output)
            

        observation = self._observe()
        reward = self._get_reward()

        print(str(self.bs.get_name()) + ": step, observation, reward: "+str(self.steps)+", "+str(self.observation)+", "+str(reward))


        self.steps = self.steps + 1
        self.done = self._is_done()


        return observation, reward, self.done, {}

    def _render(self, mode='human', close=False):

        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self):
        print("reward")

        rate_list = []
        #for i in range(10):
        #   self.sim.execute()
        rate_list.append(self.bs.get_observation())

        reward = np.mean(rate_list)

        return reward
        

    def _observe(self):
        print(str(self.bs.get_name()) + ": observe")
        
        #print("execution")
        #self.sim.execute()
        observation = self.bs.get_observation()
        print(str(self.bs.get_name()) + ": observation:" + str(observation))

        return observation


    def _is_done(self):
        print(str(self.bs.get_name()) + ": done")

        if self.steps > self.MAX_STEPS:
            print(str(self.bs.get_name()) + ": END")
            random.seed(0)
            return True
        else:
            return False


