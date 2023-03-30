import sys

import numpy as np

import math
import random

import networkx as nx

import copy


import itertools

import statistics

from tensorforce.agents import DoubleDQN
from tensorforce import Agent, Environment

import gym
from gym import wrappers
import myenv

import sys

from math import log10, floor


def calc_d(s, d): 
    ans = math.sqrt(math.pow(s[0] - d[0], 2) + math.pow(s[1] - d[1], 2))
    return(ans)
    
def d2w(x):
    return(math.pow(10, x/10.0))

def w2d(x):
    return(10 * math.log(x, 10))

def dsum(x,y):
    return(10 * math.log(math.pow(10, x/10.0) + math.pow(10, y/10.0), 10))

class Data():
    bit = 0
    placement = []

class CustomEnvironment(Environment):
 
    def __init__(self, obs_len, act_len):
        super().__init__()
        self.obs_len = obs_len
        self.act_len = act_len
 
    def states(self):
        print("states")
        return dict(type='float', shape=(self.obs_len,))
 
    def actions(self):
        print("actions")
        return dict(type='int', num_values=self.act_len)
 
    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()
 
    # Optional
    def close(self):
        print("close")
        super().close()
 
    def reset(self):
        state = np.random.random(size=(self.obs_len,))
        print("reset")
        return state
 
    #def execute(self, actions):
    #    print("execute actions:"+str(actions))
    #    #assert 0 <= actions.item() <= 3
    #    next_state = np.random.random(size=(self.obs_len,))
    #    terminal = np.random.random() < 0.5
    #    reward = np.random.random()
    #    print("execute")
    #    return next_state, terminal, reward

class BS():

    x_lim = 400
    y_lim = 400

        
    
    #config_p[0] = ['D','S','U','U','U','D','S','U','U','U']
    #config_p[1] = ['D','S','U','U','D','D','S','U','U','D']
    #config_p[2] = ['D','S','U','D','D','D','S','U','D','D']
    #config_p[3] = ['D','S','U','U','U','D','D','D','D','D']
    #config_p[4] = ['D','S','U','U','D','D','D','D','D','D']
    #config_p[5] = ['D','S','U','D','D','D','D','D','D','D']
    #config_p[6] = ['D','S','U','U','U','D','S','U','U','D']

    subframe_duration = 0.01 #s = 10ms
    
    rand_id = 0

    def __init__(self, sim, BStype, id_x, cell_color, x_pos, y_pos, cell_id, config_p):
    
        if BStype == "SBS":
            self.transmit_power_list = [40, 60, 80]
        elif BStype == "MBS":
            self.transmit_power_list = [40, 60, 80]
        else:
            sys.exit()
        
        #self.transmit_power = pow(10, self.transmit_power_list[random.randint(0,2)] / 10) * 0.001
        self.transmit_power = d2w(self.transmit_power_list[0])

        self.h = d2w(15)
        
        self.a = -2
        
        self.sim = sim
    
        #self.x_position = random.randint(0, self.x_lim)
        #self.y_position = random.randint(0, self.y_lim)

        self.x_position = x_pos
        self.y_position = y_pos


        self.BStype = BStype
        self.id = id_x
        self.cell_color = cell_color
        self.cell_id = cell_id
        
        self.config_p = config_p
        
        #self.config = self.config_p[random.randint(0,6)]
        if self.sim.mode == "conv":
            self.config = self.config_p[0]       
        elif self.sim.mode == "schD":
            self.config = self.config_p[1] 
        elif self.sim.mode == "DRL":
            self.config = self.config_p[1] 
            
        print("config:"+str(self.config))      

        #self.config_id = 3 #???
        self.config_len = len(self.config_p)
        
        print(str(self.get_name())+": "+str(self.get_xy())+": "+str(self.cell_id))
        color_chr = ["y","m","c","r","g","b","k"]
        print("plot("+str(self.get_xy()[0])+", "+str(self.get_xy()[1])+", '"+str(color_chr[self.cell_color])+"o')")

        self.state = 'D'

        self.mbs_interval = 100
        if self.get_type() == "SBS":
            self.obs_len = 12 #10
        else:
            self.obs_len = int(12 * self.mbs_interval / 10)
            
        self.observation = np.zeros(self.obs_len)
        self.actions = 0
        
        self.opt_bs = None
        
        self.inside_UE = []
        self.connecting_UE_ul = []
        self.connecting_UE_dl = []
        
        self.max_antenna_ul = 4
        self.max_antenna_dl = 4

        
            
        if self.sim.mode == "DRL":
            self.gen_bs_agent()
            
        self.throughput_ul_list = []
        self.throughput_dl_list = []

        self.c_connecting_UE_ul = []
        self.c_connecting_UE_dl = []
        self.ans_ul = []
        self.ans_ul_num = []
        self.ans_dl = []
        self.ans_dl_num = []
        
    def get_bs_throughput(self):
    
        ans1 = 0
        ans2 = 0
        
        if self.throughput_ul_list != []:
            ans1 = np.percentile(self.throughput_ul_list, 50)
        if self.throughput_dl_list != []:
            ans2 = np.percentile(self.throughput_dl_list, 50)
        
        self.throughput_ul_list = []
        self.throughput_dl_list = []
        
        return(ans1, ans2)

    def get_cell_id(self):
        return(self.cell_id)

    def gen_bs_agent(self):
    
        ENV_NAME = 'myenvtdd-v0'

        network_spec=[
            dict(type='dense', size=self.obs_len, activation='softplus'),
            dict(type='dense', size=64, activation='softplus'),
            dict(type='dense', size=32, activation='relu')
        ]
        agent_spec = 'double_dqn'
        memory = 1024; batch_size = 256; lr = 0.1; freq = 64#64
#        self.env = Environment.create(
#            environment='gym', level=ENV_NAME, max_episode_timesteps=10000,
#            sim=self.sim, bs=self, observation_len=self.obs_len, s_seed=self.id
#        )
        print("pre def env"+str(self.config_len))
        self.env = Environment.create(
            environment=CustomEnvironment, max_episode_timesteps=10,
            obs_len=self.obs_len, act_len=self.config_len
        )
        print("pre def agent")        
        self.agent = Agent.create(
            agent='double_dqn',
            states=self.env.states(),
            actions=self.env.actions(),
            memory=memory,
            batch_size=batch_size,
            exploration=0.1,#0.05,
            variable_noise=0.1,
            parallel_interactions=1,
            # learning_rate=dict(type='exponential', unit='episodes', num_steps=50, initial_value=lr, decay_rate=0.8),
            learning_rate=lr,
            # discount=1.,
            discount=0.95,
            update_frequency=freq,
            network=network_spec,
            config=dict(seed=self.id)#{'seed': 0}
        )
        print("pre env reset")
        self.env.reset()


    def set_action(self):
        
        actions = self.agent.act(states=self.observation)

        same_cell_bs = []
        for bs in self.sim.BS_list:
            if (bs.get_type() == "MBS") and (bs.get_cell_id() == self.cell_id):
                same_cell_bs.append(bs)

        if same_cell_bs != []:
            small_id = 10000
            small_id_bs = None
            for bs in same_cell_bs:
                if bs.get_id() < small_id:
                    small_id = bs.get_id()
                    small_id_bs = bs
            if small_id < self.get_id():
                actions = small_id_bs.get_action()
        
        self.actions = actions
        self.set_config(self.actions)
        print("observation:"+str(self.observation)+","+str(self.get_name())+","+str(self.sim.time_step))
        print("actions:"+str(self.actions)+","+str(self.get_name())+","+str(self.sim.time_step))

    def get_action(self):
        return(self.actions)
        
    def learning(self, reward):
        self.agent.observe(terminal=False, reward=reward)

    def find_ue_week_SINR(self):
    
        min_capacity = 100000
        min_ue = None
        for ue in (self.connecting_UE_ul + self.connecting_UE_dl):
            #print("ue:"+str(ue.get_name()))
            link = self.sim.find_link(ue, self) #??
            if link != None:
                capacity = link.get_capacity()
                if min_capacity > capacity:
                    min_capacity = capacity
                    min_ue = ue
            link = self.sim.find_link(self, ue) #??
            if link != None:
                capacity = link.get_capacity()
                if min_capacity > capacity:
                    min_capacity = capacity
                    min_ue = ue
        #print("min_ue"+str(min_ue))
        return(min_ue)

    def get_down_num(self):
        count_D = 0
        for i in self.config:
            if i == "D":
                count_D = count_D + 1
        return(count_D)

    def get_up_num(self):
        count_U = 0
        for i in self.config:
            #print("get_up_num i:"+str(i))
            if i == "U":
                count_U = count_U + 1
        return(count_U)            
        
    def reset_connecting_UE(self):
        self.connecting_UE_ul = []
        self.connecting_UE_dl = []

    def set_connecting_UE_ul(self, x):
        #print("connectin_UE:"+str(self.get_name())+", "+str(len(self.connecting_UE_ul))+", "+str(self.max_antenna_ul))
        if (len(self.connecting_UE_ul) >= self.max_antenna_ul):
            return(False)
        if (x not in self.connecting_UE_ul):
            self.connecting_UE_ul.append(x)
        return(True)

    def set_connecting_UE_dl(self, x):
        #print("connectin_UE:"+str(self.get_name())+", "+str(len(self.connecting_UE_dl))+", "+str(self.max_antenna_dl))
        if (len(self.connecting_UE_dl) >= self.max_antenna_dl):
            return(False)
        if (x not in self.connecting_UE_dl):
            self.connecting_UE_dl.append(x)
        return(True)

    def remove_connecting_UE_ul(self, x):
        if x in self.connecting_UE_ul: #??
            self.connecting_UE_ul.remove(x)
        #print("remove ul:" + str(len(self.connecting_UE_ul)))

    def remove_connecting_UE_dl(self, x):
        if x in self.connecting_UE_dl: #??
            self.connecting_UE_dl.remove(x)
        #print("remove dl:" + str(len(self.connecting_UE_dl)))

    def get_connecting_UE_ul(self):
        return(self.connecting_UE_ul)

    def get_connecting_UE_dl(self):
        return(self.connecting_UE_dl)


    def get_connecting_UE_num(self):
        return(len(self.connecting_UE_ul + self.connecting_UE_dl))

    def get_connecting_UE_name(self):
        ans = []
        for i in (self.connecting_UE_ul + self.connecting_UE_dl):
            ans.append(i.get_name())
        return(ans)
        
    def remove_inside_UE(self, x):
        if x in self.inside_UE: #??
            self.inside_UE.remove(x)

    def set_inside_UE(self, x):
        self.inside_UE.append(x)

    def get_opt_bs(self):
        return(self.opt_bs)

    def set_opt_bs(self, bs):
        self.opt_bs = bs

    def get_tpower(self):
        return(self.transmit_power)

    def get_xy(self):
        return((self.x_position, self.y_position))
        
    def set_xy(self,x):
        self.x_position=x[0]
        self.y_position=x[1]

    def get_h(self):
        return(self.h)

    def get_a(self):
        return(self.a)

    def get_generalized_xy(self):
        return((self.x_position/self.x_lim, self.y_position/self.y_lim))

    def set_config(self, x):
        self.config = self.config_p[x]
        #self.config_id = x

        #print(str(self.get_name())+": "+str(self.config))        
        
    def set_state(self, x):
        #print("state x:"+str(x))
        #print("config:"+str(self.config))
        self.state = self.config[x]
        #print("state:"+str(self.state))
        
    #def set_observation(self, x):
    #    if self.observation is None:
    #        self.observation = x
    #    else:
    #        self.observation = self.observation + x
    #    #print("set observation:"+str(self.observation))
    
    def initialize_observation(self):
        self.observation = np.zeros(self.obs_len)
    
    def get_observation(self, time_step):

        same_cell_bs = []
        for bs in self.sim.BS_list:
            if (bs.get_type() == "MBS") and (bs.get_cell_id() == self.cell_id):
                same_cell_bs.append(bs)

        if (self.get_type() == "SBS" or (time_step % self.mbs_interval == 0)):
            self.c_connecting_UE_ul = []
            self.c_connecting_UE_dl = []
            
        if same_cell_bs != []:
            for bs in same_cell_bs:
                self.c_connecting_UE_ul = self.c_connecting_UE_ul + bs.connecting_UE_ul
                self.c_connecting_UE_dl = self.c_connecting_UE_dl + bs.connecting_UE_dl
        else:
            self.c_connecting_UE_ul = self.connecting_UE_ul
            self.c_connecting_UE_dl = self.connecting_UE_dl
        
        if (self.get_type() == "SBS" or (time_step % self.mbs_interval == 0)):
            self.ans_ul = []
            self.ans_ul_num = []
        for i in self.c_connecting_UE_ul:
            for j in i.data_list:
                self.ans_ul.append(j[1])
                self.ans_ul_num.append(j[4])

        if (self.get_type() == "SBS" or (time_step % self.mbs_interval == 0)):
            self.ans_dl = []
            self.ans_dl_num = []
        for i in self.c_connecting_UE_dl:
            for j in i.data_list:
                self.ans_dl.append(j[1])
                self.ans_dl_num.append(j[4])
                
        if (self.get_type() == "SBS") or (time_step % self.mbs_interval == 0):

            #length
            lim_len = int(self.obs_len / 4)
            print("lim_len:"+str(lim_len))
        
            while len(self.ans_ul) <= lim_len:
                self.ans_ul.append(0)
            if len(self.ans_ul) > lim_len:
                self.ans_ul = self.ans_ul[0:lim_len]
                print("over ans_ul")

            while len(self.ans_ul_num) <= lim_len:
                self.ans_ul_num.append(0)
            if len(self.ans_ul_num) > lim_len:
                self.ans_ul_num = self.ans_ul_num[0:lim_len]
                print("over ans_ul_num")

            while len(self.ans_dl) <= lim_len:
                self.ans_dl.append(0)
            if len(self.ans_dl) > lim_len:
                self.ans_dl = self.ans_dl[0:lim_len]
                print("over ans_dl")

            while len(self.ans_dl_num) <= lim_len:
                self.ans_dl_num.append(0)
            if len(self.ans_dl_num) > lim_len:
                self.ans_dl_num = self.ans_dl_num[0:lim_len]
                print("over ans_dl_num")
            
            
            self.observation = self.ans_ul + self.ans_ul_num + self.ans_dl + self.ans_dl_num
        
        #return(ans)
                
    def get_type(self):
        return(self.BStype)
        
    def get_id(self):
        return(self.id)

    def get_name(self):
        return(str(self.BStype)+str(self.id))

    def get_state(self):
        return(self.state)
        
    #def get_config_id(self):
    #    return(self.config_id)
    
    def gen_random_data(self, time_step):

        #print("gen_random_data BS:"+str(self.get_name()))
        #print("gen_random_data inside_UE:"+str(self.inside_UE))
        #print("id:"+str(self.inside_UE[0].get_id())+",time_step:"+str(time_step))
        #print(self.ul_random[self.inside_UE[0].get_id()][time_step])
        #print(self.ul_random[self.inside_UE[0].get_id()])

        if self.inside_UE == []:
            return()

        
        #print("gen_random_data:"+str(self.get_name()))
        
        data_size = 0.5 * 8 #Mbyte 0.5MB * pow(10, 6) 
        #print("ul_random:"+str(self.ul_random[self.rand_id]))
        for i_env, i in enumerate(self.inside_UE):
            #print("inside_UE")
            #print("gen_random_data:"+str(i.get_name())+","+str(i.get_connecting_BS().get_name()))
            for _ in range(self.sim.ul_random[i.get_id()][time_step]):
                #print("gen_ul")
                i.gen_data(self, data_size, "ul")
                #i.gen_ul_data(self, data_size)        
            for _ in range(self.sim.dl_random[i.get_id()][time_step]):
                #print("gen_dl")
                i.gen_data(self, data_size, "dl")
                #i.gen_dl_data(self, data_size)                    
            self.rand_id = self.rand_id + 1

        
        ###################
        #self.gen_ul_data(bs, data_size)
        #self.gen_dl_data(bs, data_size)

    
    def log(self):
        i = 1
        #print(str(self.BStype) + ", id:" + str(self.id)+ ", (x,y):" + str(self.x_position) + ", " + str(self.y_position) + ", tp:"+ str(self.transmit_power) + ", h:" + str(self.h))
        
        
class MBS(BS):

    #3GPP
    bandwidth = 10 #MHz * pow(10, 6) 
#    shadowing_UE_MBS = 8 #dB
#    shadowing_UE_UE = 12 #dB
    antenna_gain_after_cable_loss = 15 #dBi
    noise_figure = 5 #dB
    TX_power = 46 #dBm (/4 = 46-6dB)



#    def UE_cell_assiciation():
#
#        for a in range(T-1):
#            f_M_DL = round((T - a) * a_M_DL)
#            f_M_UL = T - a - f_M_DL
#            f_S_dynTDD = f_M_UL + a
#            for m in range(1,M):
#                U_m_M = U_m_M_s
#                K_1[m] = K_1_s[m]
#                sorted_index = numpy.argsort(gamma)
#                sorted_U_m_M = []
#                for i in range(len(sorted_index)):
#                    sorted_U_m_M.append(U_m_M[sorted_index[i]])
#                for j in range(1, K_1[m]):
#                    calc_d_m_M_DL()
#                    calc_d_m_M_UL()


    def __init__(self, sim, BStype, id_x, cell_color, x_pos, y_pos, cell_id):
        
        if (sim.mode == "conv") or (sim.mode == "schD"):
            config_p = [[],[]]
            config_p[0] = ['U','U','U','D','D','D','D','D','D','D']
            config_p[1] = ['U','U','U','A','A','D','D','D','D','D']
        if sim.mode == "DRL":
            config_p = [[],[],[],[],[],[],[],[],[]]
            config_p[0] = ['U','D','D','D','D','D','D','D','D','D']
            config_p[1] = ['U','U','D','D','D','D','D','D','D','D']
            config_p[2] = ['U','U','U','D','D','D','D','D','D','D']
            config_p[3] = ['U','U','U','U','D','D','D','D','D','D']
            config_p[4] = ['U','U','U','U','U','D','D','D','D','D']
            config_p[5] = ['U','U','U','U','U','U','D','D','D','D']
            config_p[6] = ['U','U','U','U','U','U','U','D','D','D']
            config_p[7] = ['U','U','U','U','U','U','U','U','D','D']
            config_p[8] = ['U','U','U','U','U','U','U','U','U','D']
        

        super().__init__(sim, BStype, id_x, cell_color, x_pos, y_pos, cell_id, config_p)


    def get_TX_power(self):
        return(self.TX_power)

    def get_antenna_gain(self):
        return(self.antenna_gain_after_cable_loss)

    def get_type(self):
        return("MBS")
        
    def get_noise_figure(self):
        return(self.noise_figure)


class SBS(BS):

    #3GPP
    #TX_power = 30
    #TX_power = 27
    TX_power = 24 #dBm  (/4 = 24-6dB)
    antenna_gain = 5 #dBi
    radius = 40 #m
    noise = 13 #dB
#    shadowing_SBS_UE = 10 #dB
#    shadowing_UE_UE = 12 #dB
#    shadowing_MBS_SBS = 6 #dB
#    shadowing_SBS_SBS = 6 #dB


    def __init__(self, sim, BStype, cell_color, id_x, x_pos, y_pos, cell_id):

        if (sim.mode == "conv") or (sim.mode == "schD"):
            config_p = [[],[],[],[],[],[]]
            config_p[0] = ['U','U','U','D','D','D','D','D','D','D']
            config_p[1] = ['U','U','U','U','U','D','D','D','D','D']
            config_p[2] = ['D','U','U','U','U','D','D','D','D','D']
            config_p[3] = ['D','D','U','U','U','D','D','D','D','D']
            config_p[4] = ['D','D','D','U','U','D','D','D','D','D']
            config_p[5] = ['D','D','D','D','U','D','D','D','D','D']
        if sim.mode == "DRL":
            config_p = [[],[],[],[],[],[],[],[],[]]
            config_p[0] = ['U','D','D','D','D','D','D','D','D','D']
            config_p[1] = ['U','U','D','D','D','D','D','D','D','D']
            config_p[2] = ['U','U','U','D','D','D','D','D','D','D']
            config_p[3] = ['U','U','U','U','D','D','D','D','D','D']
            config_p[4] = ['U','U','U','U','U','D','D','D','D','D']
            config_p[5] = ['U','U','U','U','U','U','D','D','D','D']
            config_p[6] = ['U','U','U','U','U','U','U','D','D','D']
            config_p[7] = ['U','U','U','U','U','U','U','U','D','D']
            config_p[8] = ['U','U','U','U','U','U','U','U','U','D']



        super().__init__(sim, BStype, cell_color, id_x, x_pos, y_pos, cell_id, config_p)

    def get_dynamic_num(self):
        return(5)
        
    def get_connecting_ERUE_num(self):
        x = 0
        for i in (self.connecting_UE_ul + self.connecting_UE_dl):
            if i.is_ERUE == True:
                x = x + 1
        return(x)
        
    def set_up_num(self, x):
        #print("x:"+str(x))
        self.config = self.config_p[6-x]
    
    def get_TX_power(self):
        return(self.TX_power)

    def get_antenna_gain(self):
        return(self.antenna_gain)

    def get_type(self):
        return("SBS")

    def get_noise_figure(self):
        return(self.noise)


class UE():

    direction = 0
    x_grid = 100
    y_grid = 100
    x_lim = 400
    y_lim = 400

    turn_right_prob = 0.25
    turn_left_prob = 0.25
    go_strait_prob = 0.25
    
    max_datasize = 0.5
    
    #3GPP
    antenna_gain = 0 #dBi
    noise_figure = 9 #dB
    power_class = 23 #db (200 mW) (/2 = 23-3dB)
    power_control_MBS_UE = -82 #dBm
    power_control_SBS_UE = -76 #dBm
    power_control_alpha = 0.8
    
    connecting_BS = []
    

    def __init__(self, sim, UEtype, id_x, cell_color, x_pos, y_pos, bs):

        
        if UEtype == "Veh": #Vehicle
            self.speed = 9
        elif UEtype == "Ped": #Pedestrain
            self.speed = 3
        else:
            sys.exit()
        
        #self.x_axis = random.randint(0, self.x_lim)
        #self.y_axis = random.randint(0, self.y_lim)
        self.x_axis = x_pos
        self.y_axis = y_pos
        self.direction = random.randint(0, 3)
        self.sim = sim

        self.data_list = []
        
        self.id = id_x
        

        self.throughput_ul_list = []
        self.throughput_dl_list = []

        self.throughput_ul_mbs_list = []
        self.throughput_dl_mbs_list = []
        self.throughput_ul_sbs_list = []
        self.throughput_dl_sbs_list = []

        self.UEtype = UEtype
        
        self.cell_color = cell_color
        print(str(self.get_name())+": "+str(self.get_xy()))
        color_chr = ["y","m","c","r","g","b","k"]
        print("plot("+str(self.get_xy()[0])+", "+str(self.get_xy()[1])+", '"+str(color_chr[self.cell_color])+"x')")

        self.delete_link_list_ul = []
        self.delete_link_list_dl = []

        self.ue_transmit_power = 20
        self.transmit_power = d2w(self.ue_transmit_power)

        self.h = d2w(15)
        
        self.a = -2
        
        self.is_ERUE = False
        
        self.connecting_BS = []
        self.inside_BS = []

        self.set_inside_BS(bs)
        bs.set_inside_UE(self)
        
        self.interf_dl = []
        self.interf_ul = []
        

    def append_interf_dl(self, interf):
        self.interf_dl.append(interf)

    def append_interf_ul(self, interf):
        self.interf_ul.append(interf)

    def get_interf_dl(self):
        return(self.interf_dl)

    def get_interf_ul(self):
        return(self.interf_ul)

    def find_sbs_large_RSRP(self):
    
        max_RSRP = 0
        max_sbs = []
        for bs in self.sim.BS_list:
            if bs.get_type() == "SBS":
                signal = self.sim.signal(bs, self, "RSRP") #??
                if max_RSRP < signal[0]:
                    max_RSRP = signal[0]
                    max_sbs = bs
        
        return(max_sbs)
        
    def get_TX_power(self):
    
        bs = self.connecting_BS
        M = 100
        #M = 50
        #M = 1
        
        #print("bs:"+str(bs))
        if bs.get_type() ==  "MBS":
            tmp_power = 10 * math.log(M, 10) + self.power_control_MBS_UE + self.power_control_alpha * self.sim.path_loss(self, bs)
            #print("M:"+str(10 * math.log(M, 10))+"control_MBS_UE:"+str(self.power_control_MBS_UE)+"path_loss*alpha:"+str(self.power_control_alpha * self.sim.path_loss(self, bs)))
        else:
            tmp_power = 10 * math.log(M, 10) + self.power_control_SBS_UE + self.power_control_alpha * self.sim.path_loss(self, bs)
            #print("M:"+str(10 * math.log(M, 10))+"control_SBS_UE:"+str(self.power_control_SBS_UE)+"path_loss*alpha:"+str(self.power_control_alpha * self.sim.path_loss(self, bs)))
        
        #print("power class:"+str(self.power_class)+"tmp_power:"+str(tmp_power))
        power = min(self.power_class, tmp_power)
    
        return(power)

    def get_antenna_gain(self):
        return(self.antenna_gain)
        
    def get_noise_figure(self):
        return(self.noise_figure)

    def reset_connecting_BS(self):
        self.connecting_BS = []

    def set_connecting_BS(self, x):
        self.connecting_BS = x

    def remove_connecting_BS(self, x):
        self.connecting_BS = []

    def set_inside_BS(self, x):
        self.inside_BS = x

    def remove_inside_BS(self, x):
        self.inside_BS = []

    def get_connecting_BS(self):
        return(self.connecting_BS)
    
    def get_tpower(self):
        return(self.transmit_power)
        
    def get_h(self):
        return(self.h)

    def get_a(self):
        return(self.a)
        
    def get_xy(self):
        return((self.x_axis, self.y_axis))

    def set_xy(self,x):
        self.x_axis=x[0]
        self.y_axis=x[1]
        
    def get_generalized_xy(self):
        return((self.x_axis/self.x_lim, self.y_axis/self.y_lim))

    def get_data_list(self):
        return(self.data_list)

    def get_id(self):
        return(self.id)

    def get_name(self):
        return(str(self.UEtype)+str(self.id))
        
    def get_type(self):
        return("UE")
            
    def gen_data(self, bs, data_size, direction): #-> Data
        if direction == "ul":
            self.data_list.append([bs, data_size, data_size, 0, 0, 'ul'])
        else:
            self.data_list.append([bs, data_size, data_size, 0, 0, 'dl'])
        
    
    def renew_time_stamp(self):
        for i in self.data_list:
            if i[3] == 1:
                i[4] = i[4] + 1
    
    def gen_link(self, s, d): #-> Link
        
        e_link = self.find_link(s, d)
        if e_link != None:
            #print("found")
            return(e_link)
        #print("notfound")

        if s.get_type() ==  "MBS" or s.get_type() == "SBS":
            connected = s.set_connecting_UE_dl(d)
            #print("connected dl:"+str(s.get_name())+", "+str(connected))
        else:
            connected = d.set_connecting_UE_ul(s)
            #print("connected ul:"+str(d.get_name())+", "+str(connected))


        if connected == True:
            link = Link(self.sim, s, d)
            self.sim.add_link(link)
            link.get_ue().set_connecting_BS(link.get_bs())
            #print("gen_link: bs:"+str(link.get_bs().get_name()))

            return(link)
        else:
            return(None)

    def find_link(self, s, d): #-> Link
        #print("find_link:"+str(self.sim.find_link(s, d)))
        return(self.sim.find_link(s, d))
        
    def delete_link_ul(self, i): #-> Link
        
        self.sim.del_link_ul(i)

    def delete_link_dl(self, i): #-> Link
        
        self.sim.del_link_dl(i)              
            
    def send(self, slot_id):

        for i_env, (bs, data_size, start_bool, _, _, direction) in enumerate(self.data_list):
            if direction == "ul":
                ##print(str(bs.get_name())+","+str(data_size)+","+str(x)+","+str(y))
                up_link = self.find_link(self, bs)
                if up_link != None:
                    ##print("find link"+str(bs.get_connecting_UE_name()))
                    if bs.get_state() == 'U':
                        capacity = up_link.get_capacity()
                        self.data_list[i_env][3] = 1
                        #print("uplink capacity, bs:" + str(capacity)+","+str(bs.get_name()))
                        #print("bs state:" + str(bs.get_state()))
                        #print("ul data before:" + str(self.data_list))
                        if self.data_list[i_env][1] > capacity:
                            self.data_list[i_env][1] = self.data_list[i_env][1] - capacity
                            #print("send:"+str(self.data_list[i_env][1])+", "+str(capacity))
                        else:
                            self.data_list[i_env][1] = 0
                        if self.data_list[i_env][1] == 0:
                            self.calc_UE_throughput(self.data_list[i_env], "ul")
                            self.data_list.pop(i_env)
                            self.delete_link_list_ul.append(up_link)
                        elif  self.data_list[i_env][4] > 800:
                            self.data_list.pop(i_env)
                            self.delete_link_list_ul.append(up_link)


    def recieve(self, slot_id):

        for i_env, (bs, data_size, start_bool, _, _, direction) in enumerate(self.data_list):
            if direction == "dl":
                dl_link = self.find_link(bs, self)
                if dl_link != None:
                    if bs.get_state() == 'D':
                        capacity = dl_link.get_capacity()
                        self.data_list[i_env][3] = 1
                        #print("downlink capacity, bs:" + str(capacity)+","+str(bs.get_name()))
                        #print("bs state:" + str(bs.get_state()))
                        #print("dl data before:" + str(self.data_list))
                        if self.data_list[i_env][1] > capacity:
                            self.data_list[i_env][1] = self.data_list[i_env][1] - capacity
                            #print("recieve:"+str(self.data_list[i_env][1])+", "+str(capacity))
                        else:
                            self.data_list[i_env][1] = 0
                        if self.data_list[i_env][1] == 0:
                            #print("data list:"+str(self.data_list[i_env]))
                            self.calc_UE_throughput(self.data_list[i_env], "dl")
                            self.data_list.pop(i_env)
                            self.delete_link_list_dl.append(dl_link)
                        elif  self.data_list[i_env][4] > 800:
                            self.data_list.pop(i_env)
                            self.delete_link_list_dl.append(dl_link)
                #print("dl data after:" + str(self.data_list))

    def calc_UE_throughput(self, x, y):
        bs = x[0]
        data_size = x[2]
        duration = x[4]
        if duration == 0:
            throughput = 0
        else:
            throughput = (data_size * 1.0) / (duration * 0.01)  #0.01 #s = 10ms
        #print("data_size:"+str(data_size)+"duration:"+str(duration)+"throughput:"+str(throughput))
        
        if y == "ul":
            self.sim.throughput_ul_list.append(throughput)
            self.throughput_ul_list.append(throughput)
            bs.throughput_ul_list.append(throughput)
            if x[0].get_type() == "MBS":
                self.sim.throughput_ul_mbs_list.append(throughput)
                self.throughput_ul_mbs_list.append(throughput)
            else:
                self.sim.throughput_ul_sbs_list.append(throughput)
                self.throughput_ul_sbs_list.append(throughput)
        else:
            self.sim.throughput_dl_list.append(throughput)
            self.throughput_dl_list.append(throughput)
            bs.throughput_dl_list.append(throughput)
            if x[0].get_type() == "MBS":
                self.sim.throughput_dl_mbs_list.append(throughput)
                self.throughput_dl_mbs_list.append(throughput)
            else:
                self.sim.throughput_dl_sbs_list.append(throughput)
                self.throughput_dl_sbs_list.append(throughput)

            
        #print("throughput_ul:"+str(self.throughput_ul_list))
        #print("throughput_dl:"+str(self.throughput_dl_list))
        
    def get_UE_throughput(self):
    
        ans1 = 0
        ans2 = 0
        
        if self.throughput_ul_list != []:
            ans1 = np.percentile(self.throughput_ul_list, 50)
        if self.throughput_dl_list != []:
            ans2 = np.percentile(self.throughput_dl_list, 50)
            
        return(ans1, ans2)
        
    def reset_UE_throughput(self):

        self.throughput_ul_list = []
        self.throughput_dl_list = []
    
    def clear_links_ul(self):
        for i in self.delete_link_list_ul:
            self.delete_link_ul(i)
        self.delete_link_list_ul = []

    def clear_links_dl(self):
        for i in self.delete_link_list_dl:
            self.delete_link_dl(i)
        self.delete_link_list_dl = []
    
    def move(self):
        #print("move:"+str(self.direction))
        #print(self.get_xy())
        self.move_t(self.speed)
    
    def move_t(self, n): #0:up, 1:right, 2:down, 3:left
        #print("move_t:"+str(self.direction))
        if self.direction == 0:
            self.moving_up(n)
        elif self.direction == 1:
            self.moving_right(n)
        elif self.direction == 2:
            self.moving_down(n)
        else:
            self.moving_left(n)       
    
    def moving_up(self, n): #0
        pre_y_axis = self.y_axis
        self.y_axis = self.y_axis + n
        flag = False
        if self.y_axis >= self.y_lim:
            flag = True
        if (pre_y_axis == self.y_lim) or (((pre_y_axis % self.y_grid) != 0) and ((pre_y_axis // self.y_grid) != (self.y_axis // self.y_grid))):
            self.coner(flag)
            tmp_y_axis = self.y_axis
            self.y_axis = (self.y_axis // self.y_grid) * self.y_grid
            #print("tmp_y_axis"+str(tmp_y_axis) + "self.y_axis:"+str(self.y_axis))
            self.move_t(tmp_y_axis - self.y_axis)
    
    def moving_down(self, n): #2
        pre_y_axis = self.y_axis
        self.y_axis = self.y_axis - n       
        flag = False
        if self.y_axis <= 0:
            flag = True
        if (pre_y_axis == 0) or (((pre_y_axis % self.y_grid) != 0) and ((pre_y_axis // self.y_grid) != (self.y_axis // self.y_grid))):
            self.coner(flag)
            tmp_y_axis = self.y_axis
            self.y_axis = (pre_y_axis // self.y_grid) * self.y_grid
            #print("self.y_axis:"+str(self.y_axis) + "tmp_y_axis"+str(tmp_y_axis))
            self.move_t(self.y_axis - tmp_y_axis)
            
    def moving_right(self, n): #1
        pre_x_axis = self.x_axis
        self.x_axis = self.x_axis + n
        flag = False
        if self.x_axis >= self.x_lim:
            flag = True
        if (pre_x_axis == self.x_lim) or (((pre_x_axis % self.x_grid) != 0) and ((pre_x_axis // self.x_grid) != (self.x_axis // self.x_grid))):
            self.coner(flag)
            tmp_x_axis = self.x_axis
            self.x_axis = (self.x_axis // self.x_grid) * self.x_grid
            #print("tmp_x_axis"+str(tmp_x_axis) + "self.x_axis:"+str(self.x_axis))
            self.move_t(tmp_x_axis - self.x_axis)
    
    def moving_left(self, n): #3
        pre_x_axis = self.x_axis
        self.x_axis = self.x_axis - n
        flag = False
        if self.x_axis <= 0:
            flag = True
            #print("flag:True")
        if (pre_x_axis == 0) or (((pre_x_axis % self.x_grid) != 0) and ((pre_x_axis // self.x_grid) != (self.x_axis // self.x_grid))):
            self.coner(flag)
            tmp_x_axis = self.x_axis
            self.x_axis = (pre_x_axis // self.x_grid) * self.x_grid
            #print("self.x_axis:"+str(self.x_axis) + "tmp_x_axis"+str(tmp_x_axis))
            self.move_t(self.x_axis - tmp_x_axis)
            
    def coner(self, flag):
        if flag == True:
            prob = random.random() * (self.turn_right_prob + self.turn_left_prob)
            #print("prob:"+str(prob))
        else:
            prob = random.random()
        if self.turn_right_prob >= prob:
            self.direction = (self.direction + 1) % 4
        elif self.turn_right_prob + self.turn_left_prob >= prob:
            self.direction = (self.direction - 1) % 4

    def log(self):
        i=1
        #print(str(self.UEtype) + ", id:" + str(self.id)+ ", (x,y):" + str(self.x_axis) +','+ str(self.y_axis) + ", direction:"+ str(self.direction) + ", speed:" + str(self.speed) + str(self.data_list))



class Link():

    def __init__(self, sim, s, d):
        self.sim = sim
        self.source = s
        self.distination = d
        self.capacity = 0

    def get_d(self):
        return(self.distination)
        
    def get_s(self):
        return(self.source)

    def get_bs(self):
        if self.source.get_type() == "SBS" or self.source.get_type() == "MBS":
            return(self.source)
        else:
            return(self.distination)

    def change_bs(self, x):
        #print("change_bs before:"+str(self.source.get_name()) +", " +str(self.distination.get_name()))
        if self.source.get_type() == "SBS" or self.source.get_type() == "MBS":
            self.source = x
        else:
            self.distination = x
        #print("change_bs after:"+str(self.source.get_name()) +", " +str(self.distination.get_name()))
        
    def get_ue(self):
        if self.source.get_type() == "SBS" or self.source.get_type() == "MBS":
            return(self.distination)
        else:
            return(self.source)
        
    def get_distance(self):
        return(calc_d(self.get_d().get_xy(), self.get_s().get_xy()))
        
    def set_capacity(self, c):
        self.capacity = c

    def get_capacity(self):
        #print("capacity:"+str(self.capacity)+", "+str(self.get_bs().get_name())+", "+str(self.get_ue().get_name()))
        return(self.capacity)
        
    def is_active(self):
        #print("is_active: sender:"+str(self.get_s().get_name())+"reciever:"+str(self.get_d().get_name()))

        if (self.get_s().get_type() == "SBS") or (self.get_s().get_type() == "MBS"):
            #print("config id D"+str(self.get_s().get_config_id()))
            if self.get_s().get_state() == "D":
                return(True)
            else:
                return(False)
        else:
            #print("config id U"+str(self.get_d().get_config_id()))
            if self.get_d().get_state() == "U":
                return(True)
            else:
                return(False)
        

    #def log(self):
        #print("(s,d):" + str(self.source) + str(self.distination) + ", capacity:" + str(self.capacity) + ", rate:"+ str(self.rate))


class Simulator():

    h = 1.0
    alpha = 1.0
    sigma = 1.0
    bandwidth = 10.0 # 10MHz

    def __init__(self, mode):
    
        
        self.Link_list = []
        self.BS_list = []
        self.UE_list = []
        
        #self.fig = plt.figure(figsize=(10,10))
        
        self.time_slot = 0
        self.time_step = 0

        self.throughput_ul_list = []
        self.throughput_dl_list = []
        self.throughput_ul_mbs_list = []
        self.throughput_dl_mbs_list = []
        self.throughput_ul_sbs_list = []
        self.throughput_dl_sbs_list = []
        
        self.mode = mode
        
        self.arrival_rate_list = [0.005, 0.003, 0.005] #[0.001, 0.003, 0.005] #[0.1, 0.3, 0.5] #0.1, 0.01
        self.arrival_rate = self.arrival_rate_list[0]
        
        #self.ul_random = np.random.poisson(self.arrival_rate, 100000) ##max_num self.arrival_rate * self.subframe_duration
        #self.dl_random = np.random.poisson(self.arrival_rate, 100000) ##max_num
        
        self.ul_random = np.random.poisson(self.arrival_rate/2, (1000, 11000)) ##max_num self.arrival_rate * self.subframe_duration
        self.dl_random = np.random.poisson(self.arrival_rate, (1000, 11000)) ##max_num

        #self.ul_random = np.zeros((1000, 10000), dtype=int)
        #self.ul_exp = np.random.exponential(1/(self.arrival_rate/2),(1000, 10))
        #for i in range(self.ul_exp.shape[0]):
        #    index = 0
        #    for j in range(self.ul_exp.shape[1]):
        #        index = index + int(self.ul_exp[i][j])
        #        #print(index)
        #        if index < self.ul_random.shape[1]:
        #            self.ul_random[i][index] = 1

        
        #self.dl_random = np.zeros((1000, 10000), dtype=int)
        #self.dl_exp = np.random.exponential(1/self.arrival_rate,(1000, 10))
        #print(self.dl_exp)
        #for i in range(self.dl_exp.shape[0]):
        #    index = 0
        #    for j in range(self.dl_exp.shape[1]):
        #        index = index + int(self.dl_exp[i][j])
        #        if index < self.dl_random.shape[1]:
        #            self.dl_random[i][index] = 1
                
        #print("len:"+str(self.dl_random.shape[0]))
        #for i in range(self.dl_random.shape[0]):
        #    print("dl_random:"+str(sum(self.dl_random[i][0:999])))

        #print("len:"+str(self.ul_random.shape[0]))
        #for i in range(self.ul_random.shape[0]):
        #    print("ul_random:"+str(sum(self.ul_random[i][0:999])))

        

#    def set_send_recieve(self):
#    
#        #print("AAA3")
#        env = 0
#        flag = 0
#        while (flag == 0):
#            flag = 1
#            #print("AAA2")
#            for i in self.UE_list:
#                #print("AAA1")
#                data_list = i.get_data_list()
#                if len(data_list) > env:
#                    #print("AAA")
#                    flag = flag * 0
#                    bs, _, _, _, _, direction = data_list[env]
#                    if direction == "ul":
#                        i.gen_link(i, bs)
#                    if direction == "dl":
#                        i.gen_link(bs, i)
#                
#            env = env + 1        

    def set_send_recieve(self):
    
        ##print("AAA3")
        #env = 0
        #flag = 0
        #while (flag == 0):
        #    flag = 1
        #    #print("AAA2")
        #    for i in self.UE_list:
        #        #print("AAA1")
        #        data_list = i.get_data_list()
        #        if len(data_list) > env:
        #            #print("AAA")
        #            flag = flag * 0
        #            bs, _, _, _, _, direction = data_list[env]
        #            if direction == "ul":
        #                i.gen_link(i, bs)
        #            if direction == "dl":
        #                i.gen_link(bs, i)
        #        
        #    env = env + 1        

        for i in self.UE_list:
            data_list = i.get_data_list()
            for bs, _, _, flag, _, direction in data_list:
                if flag == 1:
                    if direction == "ul":
                        i.gen_link(i, bs)
                    if direction == "dl":
                        i.gen_link(bs, i)

        for i in self.UE_list:
            data_list = i.get_data_list()
            for bs, _, _, flag, _, direction in data_list:
                if flag == 0:
                    if direction == "ul":
                        i.gen_link(i, bs)
                    if direction == "dl":
                        i.gen_link(bs, i)

    def round_to_1(self, x):
        #print("x:"+str(x))
        if x != 0:
            return round(x, -int(floor(log10(abs(x))))+1)
        else:
            return 0

    def print_interf(self, output_f):
    
        fxy = open(output_f+"_xy.txt", mode = 'w')
        ful = open(output_f+"_interf_ul.txt", mode = 'w')
        fdl = open(output_f+"_interf_dl.txt", mode = 'w')
        for i in self.UE_list:
            xy = i.get_xy()
            ul = i.get_interf_ul()
            dl = i.get_interf_dl()
            fxy.write(str(xy[0])+" "+str(xy[1])+"\n")
            for j in ul:
                ful.write(str(self.round_to_1(j))+" ")
            ful.write("\n")
            for j in dl:
                fdl.write(str(self.round_to_1(j))+" ")
            fdl.write("\n")

    def print_buffer_num(self):

        num_buffer = 0
        #num_ul_buffer = 0
        #num_dl_buffer = 0
        for i in self.UE_list:
            for j in i.get_data_list():
                if j[3] == 0:
                    num_buffer = num_buffer + len(i.get_data_list())
                    print("buffer:"+str(i.get_name())+","+str(j[0].get_name())+","+str(j[3])+","+str(j[4]))

        print("buffer:"+str(num_buffer))

    def sum_UE_throughput(self):

        ans1_1 = 0
        ans1_2 = 0
        ans1_3 = 0
        
        ans2_1 = 0
        ans2_2 = 0
        ans2_3 = 0
        
        ans3_1 = 0
        ans3_2 = 0
        ans3_3 = 0
        
        ans4_1 = 0
        ans4_2 = 0
        ans4_3 = 0
        
        ans5_1 = 0
        ans5_2 = 0
        ans5_3 = 0
        
        ans6_1 = 0
        ans6_2 = 0
        ans6_3 = 0
        
        #print("throughput_ul_list:"+str(self.throughput_ul_list))
        #print("throughput_dl_list:"+str(self.throughput_dl_list))
        
        if self.throughput_ul_list != []:
            #ans1 = sum(self.throughput_ul_list)
            ans1_1 = np.percentile(self.throughput_ul_list, 95)
            ans1_2 = np.percentile(self.throughput_ul_list, 50)
            ans1_3 = np.percentile(self.throughput_ul_list, 5)
        if self.throughput_dl_list != []:
            #ans2 = sum(self.throughput_dl_list)
            ans2_1 = np.percentile(self.throughput_dl_list, 95)
            ans2_2 = np.percentile(self.throughput_dl_list, 50)
            ans2_3 = np.percentile(self.throughput_dl_list, 5)

        if self.throughput_ul_mbs_list != []:
            #ans3 = sum(self.throughput_ul_mbs_list)
            ans3_1 = np.percentile(self.throughput_ul_mbs_list, 95)
            ans3_2 = np.percentile(self.throughput_ul_mbs_list, 50)
            ans3_3 = np.percentile(self.throughput_ul_mbs_list, 5)
        if self.throughput_dl_mbs_list != []:
            #ans4 = sum(self.throughput_dl_mbs_list)
            ans4_1 = np.percentile(self.throughput_dl_mbs_list, 95)
            ans4_2 = np.percentile(self.throughput_dl_mbs_list, 50)
            ans4_3 = np.percentile(self.throughput_dl_mbs_list, 5)

        if self.throughput_ul_sbs_list != []:
            #ans5 = sum(self.throughput_ul_sbs_list)
            ans5_1 = np.percentile(self.throughput_ul_sbs_list, 95)
            ans5_2 = np.percentile(self.throughput_ul_sbs_list, 50)
            ans5_3 = np.percentile(self.throughput_ul_sbs_list, 5)
        if self.throughput_dl_sbs_list != []:
            #ans6 = sum(self.throughput_dl_sbs_list)
            ans6_1 = np.percentile(self.throughput_dl_sbs_list, 95)
            ans6_2 = np.percentile(self.throughput_dl_sbs_list, 50)
            ans6_3 = np.percentile(self.throughput_dl_sbs_list, 5)
            
            
        #self.throughput_ul_list = []
        #self.throughput_dl_list = []

        return(ans1_1, ans1_2, ans1_3, ans2_1, ans2_2, ans2_3, ans3_1, ans3_2, ans3_3, ans4_1, ans4_2, ans4_3, ans5_1, ans5_2, ans5_3, ans6_1, ans6_2, ans6_3)

    def add_ue(self, UEtype, id_x, cell_color, x, y, bs):
        ue = UE(self, UEtype, id_x, cell_color, x, y, bs)
        self.UE_list.append(ue)
        return(ue)
    
    def add_bs(self, BStype, id_x, cell_color, x, y, cell_id):
        if BStype == "MBS":
            bs = MBS(self, BStype, id_x, cell_color, x, y, cell_id)
        elif BStype == "SBS":
            bs = SBS(self, BStype, id_x, cell_color, x, y, cell_id)        
        self.BS_list.append(bs)
        return(bs)
        
    def add_bs_agent(self, BStype, id_x, cell_color, x, y, cell_id):
        #print("add_bs_agent")
        if BStype == "MBS":
            bs = MBS(self, BStype, id_x, cell_color, x, y, cell_id)
        elif BStype == "SBS":
            bs = SBS(self, BStype, id_x, cell_color, x, y, cell_id)        
        self.BS_list.append(bs)
        #bs.gen_bs_agent()
        return(bs)
    
    def add_link(self, x):
        self.Link_list.append(x)
        #print("add_link:"+str(x.get_s().get_name())+", "+str(x.get_d().get_name()))
        
    def del_link_ul(self, x):
    
        #print("del_link:"+str(x.get_s().get_name())+", "+str(x.get_d().get_name()))
        
        #for i in self.Link_list:
        #    print("Link_list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name()))
            
    
        x.get_bs().remove_connecting_UE_ul(x.get_ue())
        exist = False
        for i in self.Link_list:
            if (i != x) and (i.get_s() == x.get_d()) and (i.get_d() == x.get_s()):
                exist = True
        if exist == False:
            #print("remove ul")
            x.get_ue().remove_connecting_BS(x.get_bs())
        #    print("del_link:"+str(x.get_bs().get_name())+","+str(x.get_bs().get_connecting_UE_name()))
            
        for i_env, i in enumerate(self.Link_list):
            if i == x:
                self.Link_list.pop(i_env)
         #       print("after")
         #       for i in self.Link_list:
         #           print("Link_list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name()))


                break

    def del_link_dl(self, x):
    
        #print("del_link:"+str(x.get_s().get_name())+", "+str(x.get_d().get_name()))
        
        #for i in self.Link_list:
        #    print("Link_list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name()))
            
    
        x.get_bs().remove_connecting_UE_dl(x.get_ue())
        exist = False
        for i in self.Link_list:
            if (i != x) and (i.get_s() == x.get_d()) and (i.get_d() == x.get_s()):
                exist = True
        if exist == False:
            print("remove dl")
            x.get_ue().remove_connecting_BS(x.get_bs())
        #    print("del_link:"+str(x.get_bs().get_name())+","+str(x.get_bs().get_connecting_UE_name()))
            
        for i_env, i in enumerate(self.Link_list):
            if i == x:
                self.Link_list.pop(i_env)
         #       print("after")
         #       for i in self.Link_list:
         #           print("Link_list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name()))


                break

    def get_bs_list(self):
        return(self.BS_list)

    def get_ue_list(self):
        return(self.UE_list)

    def get_link_list(self):
        return(self.Link_list)

    def get_dl_list(self):
        return(self.DL_list)

    def log(self):
        #print(self.BS_list)
        #print(self.UE_list)
        #for i in self.BS_list:
        #    i.log()
        for i in self.UE_list:
            i.log()

        ####server only
        #if self.UE_list != []:
        #    self.visualization()

    def get_bs_num(self):
        return(len(self.BS_list))

    def get_ue_num(self):
        return(len(self.UE_list))

    def find_link(self, s, d):
        #print("find_list:"+str(self.Link_list))
        for i in self.Link_list:
           if i.get_s() == s and i.get_d() == d:
               return(i)
        return(None)


    def path_loss(self, s, r): #3GPP, LOS, d in km

        d = calc_d(s.get_xy(), r.get_xy()) / 1000
        #print("r:"+str(r)+"sender:"+str(s.get_name())+"reciever:"+str(r.get_name()))
        #print("sender:"+str(s.get_xy())+"reciever:"+str(r.get_xy()))
        #print("distance:"+str(d))
        
        s_t = s.get_type()
        r_t = r.get_type()
        #print("s_t:"+str(s_t)+", r_t:"+str(r_t))
        #print("sender:"+str(s.get_name()))
        #print("reciever::"+str(r.get_name()))
        
        if (s_t == "MBS" and r_t == "SBS") or (s_t == "SBS" and r_t == "MBS"):
            ans = 100.7 + 23.5 * math.log(d, 10)
            #ans = 125.2+36.3 * math.log(d, 10) #NLOS
        elif (s_t == "SBS" and r_t == "SBS"):
            if d < 2/3:
                ans = 98.4+20 * math.log(d, 10)
            else:
                ans = 101.9+40 * math.log(d, 10)
            #ans = 169.36 + 40 * math.log(d, 10) #NLOS
        elif (s_t == "MBS" and r_t == "MBS"):
            ans = 98.45+20 * math.log(d, 10) 
        elif (s_t == "MBS" and r_t == "UE") or (s_t == "UE" and r_t == "MBS"):
            #print("MBS-UE")
            ans = 103.4+24.2 * math.log(d, 10)
            #ans = 131.1+42.8 * math.log(d, 10) #NLOS
        elif (s_t == "SBS" and r_t == "UE") or (s_t == "UE" and r_t == "SBS"):
            #print("SBS-UE")
            ans = 103.8 + 20.9 * math.log(d, 10)
            #ans = 145.4+37.5 * math.log(d, 10) #NLOS
        elif (s_t == "UE" and r_t == "UE"):
            if d <= 0.05:
                ans = 98.45+20 * math.log(d, 10)
            else:
                ans = 40 * math.log(d, 10) + 175.78 

        #if (s_t == "MBS" or s_t == "SBS"):
        #    ue = r
        #else:
        #    ue = s

        #if ue.get_connecting_BS() == "MBS":
        #    ans = 103.4+24.2 * math.log(d, 10)
        #else:
        #    ans = 103.8 + 20.9 * math.log(d, 10)
        
        #print("path loss:"+str(ans))
                
        return(ans)
                


    def shadowing(self, s, r):
        shadowing_MBS_MBS = 8 #dB
        shadowing_UE_MBS = 8 #dB
        shadowing_SBS_UE = 10 #dB
        shadowing_UE_UE = 12 #dB
        shadowing_MBS_SBS = 6 #dB
        shadowing_SBS_SBS = 6 #dB

        if ((s.get_type() == "MBS") and (r.get_type() == "MBS")) or ((s.get_type() == "MBS") and (r.get_type() == "MBS")):
            return(shadowing_MBS_MBS)
        if ((s.get_type() == "UE") and (r.get_type() == "MBS")) or ((s.get_type() == "MBS") and (r.get_type() == "UE")):
            return(shadowing_UE_MBS)
        if ((s.get_type() == "SBS") and (r.get_type() == "UE")) or ((s.get_type() == "UE") and (r.get_type() == "SBS")):
            return(shadowing_SBS_UE)
        if ((s.get_type() == "UE") and (r.get_type() == "UE")) or ((s.get_type() == "UE") and (r.get_type() == "UE")):
            return(shadowing_UE_UE)
        if ((s.get_type() == "MBS") and (r.get_type() == "SBS")) or ((s.get_type() == "SBS") and (r.get_type() == "MBS")):
            return(shadowing_MBS_SBS)
        if ((s.get_type() == "SBS") and (r.get_type() == "SBS")) or ((s.get_type() == "SBS") and (r.get_type() == "SBS")):
            return(shadowing_SBS_SBS)

        #s_t = s.get_type()
        #r_t = r.get_type()

        #if (s_t == "MBS" or s_t == "SBS"):
        #    ue = r
        #else:
        #    ue = s

        #if ue.get_connecting_BS() == "MBS":
        #    return(shadowing_UE_MBS)
        #else:
        #    return(shadowing_SBS_UE)
        

    def signal(self, s, d, string):

        #print("string:"+str(string))
        ans1 = s.get_TX_power()
        ans2 = d.get_antenna_gain()
        ans3 = self.path_loss(s, d)
        ans4 = self.shadowing(s, d)
        ans5 = d.get_noise_figure()
        #print("TX_power:"+str(ans1))
        #print("antenna gain db:"+str(ans2))
        #print("antenna gain w:"+str(d2w(ans2)))
        #print("path loss db:"+str(ans3))
        #print("path loss w:"+str(d2w(ans3)))
        #print("shadowing db:"+str(ans4))
        #print("shadowing w:"+str(d2w(ans4)))
        #print("noise figure:"+str(ans5))

        ans = ans1 + ans2 - ans5 - ans4 - ans3
        #print(str(s.get_name())+","+str(d.get_name())+", power, gain, noise, pathl, shad, signal, dis:"+str(ans1)+","+str(ans2)+","+str(ans5)+","+str(ans4)+","+str(ans3)+","+str(ans)+","+str(calc_d(s.get_xy(), d.get_xy())))
        #print("signal db:" + str(ans))
        #ans = s.get_TX_power() * d2w(s.get_antenna_gain()) * d2w(self.path_loss(s, d)) * d2w(self.shadowing(s, d))
        #http://www.rf-world.jp/bn/RFW09/samples/p013-014.pdf
        #http://www.mobile.ecei.tohoku.ac.jp/lecture/wireless/wireless_05.pdf
        #if string == "interf":
        #print(string+":"+str(ans1)+","+str(ans2)+","+str(ans5)+","+str(ans4)+","+str(ans3)+","+str(ans)+","+str(calc_d(s.get_xy(), d.get_xy())))
        #print(string+":"+str(ans))


        
        return(d2w(ans), ans)
        

    def capacity_calculation(self):

        #3GPP
        
#        signal(sender, reciever) = sender.get_TX_power() * d2w(sender.get_antenna_gain()) * d2w(path_loss(sender, reciever) * d2w(shadowing(sender, reciever))
     
#        sinr_upper = signal(s1, r1)
        
#        interf = 0
#        for s in all_signals:
#            interf += signal(s, r1)

#        sinr_bottom = interf + d2w(reciever.noise(r1))

        #http://www.mobile.ecei.tohoku.ac.jp/intro/pdf/07_digital.pdf
        
        #need signal list up
        
        #round robin
        #proportional fairness
        
        #lambda poassion?
        #UPT

        ######

        noise_floor = -108  #dBm
        #Presentation of Specification to TSG or WG - 3GPP

        num_SINR = 0
                
        for i in self.Link_list:
            #print("Link_list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name())+", "+str(i.is_active()))
            if i.is_active() == True:
                sinr_upper, sinr_upper_db = self.signal(i.get_s(), i.get_d(), 'upper')
                #print("tmp:"+str(w2d(sinr_upper))+", "+str(sinr_upper_db))

                #print("i:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name())+", "+str(sinr_upper_db))
                interf = 0
                interf_db = None
                interf_bs_list = [i.get_bs()]
                interf_bs_cell_list = [i.get_bs().get_cell_id()]
                for j in self.Link_list:
                    #if (j != i) and (j.is_active() == True) and (j.get_bs().get_cell_id() != i.get_bs().get_cell_id()) and (j.get_bs() not in interf_bs_list): #???
                    #if (j != i) and (j.is_active() == True) and (j.get_bs().get_cell_id() != i.get_bs().get_cell_id()): #???
                    if (j != i) and (j.is_active() == True) and (j.get_bs() not in interf_bs_list) and (j.get_bs().get_cell_id() not in interf_bs_cell_list): #???
                        interf_tmp, interf_tmp_db = self.signal(j.get_s(), i.get_d(), 'interf')
                        #print("j:"+str(j.get_s().get_name())+", "+str(j.get_d().get_name())+", "+str(interf_tmp_db))
                        interf += interf_tmp
                        if interf_db == None:
                            interf_db = interf_tmp_db
                        else:
                            interf_db = dsum(interf_db, interf_tmp_db)
                        interf_bs_list.append(j.get_bs())
                        interf_bs_cell_list.append(j.get_bs().get_cell_id())

                if i.get_s().get_type() == "SBS" or i.get_s().get_type() == "MBS":
                    i.get_d().append_interf_dl(interf)
                else:
                    i.get_s().append_interf_ul(interf)

                #print("sinr_upper:"+str(sinr_upper))
                #print("sinr_interf:"+str(interf))
                #noise_tmp_db = noise_floor + i.get_d().get_noise_figure()
                noise_tmp_db = noise_floor
                #print("noise_tmp1:"+str(noise_tmp1))            
                #print("noise_tmp2:"+str(d2w(noise_tmp1)))            
                #noise_tmp = math.pow(d2w(noise_tmp1), 2)
                noise_tmp = d2w(noise_tmp_db)
                #print("noise_tmp:"+str(noise_tmp))
                sinr_bottom = interf + noise_tmp
                if interf_db == None:
                    sinr_bottom_db = noise_tmp_db
                else:
                    sinr_bottom_db = dsum(interf_db, noise_tmp_db)

                sinr = sinr_upper / sinr_bottom
                sinr_db = sinr_upper_db - sinr_bottom_db
                #print("SINR db:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name())+", "+str(w2d(sinr_upper))+", "+str(w2d(sinr_bottom))+", "+str(w2d(sinr))+", "+str(sinr_upper_db)+", "+str(sinr_bottom_db)+", "+str(sinr_db))
                #print("SINR db:"+str(i.get_s().get_type())+", "+str(i.get_d().get_type())+", "+str(w2d(sinr_upper))+", "+str(w2d(sinr_bottom))+", "+str(w2d(sinr)))
                print("SINR db:"+str(i.get_s().get_type())+", "+str(i.get_d().get_type())+", "+str(w2d(sinr)))

                num_SINR = num_SINR + 1
                
                capacity = self.bandwidth * math.log((1 + sinr), 2)

                capacity = 0.01 * 0.1 * capacity #10ms 1subframe
                #print("capacity:"+str(capacity))
                
                i.set_capacity(capacity)
            else:
                i.set_capacity(0)

        print("num_SINR:"+str(self.time_slot)+", "+str(num_SINR))
                

#            if sinr_bottom != 1.0:
#                print("UL sinr_interf, slot, bs, ue capacity: 1, "+str(self.time_slot)+","+str(i.get_bs().get_name())+","+str(i.get_ue().get_name())+","+str(capacity))
#            else:
#                print("UL sinr_interf, slot, bs, ue capacity: 0, "+str(self.time_slot)+","+str(i.get_bs().get_name())+","+str(i.get_ue().get_name())+","+str(capacity))
                



    def execute_DRL(self):


        for i in self.BS_list:
            if i.get_type() == "SBS":
                #i.initialize_observation()
                i.set_action()
            else:
                if self.time_step % i.mbs_interval == 0:
                    i.set_action()

                   
        #opt_bs_num = (self.time_step // 10) % len(self.BS_list)
        #for i in self.BS_list:
        #    i.set_opt_bs(self.BS_list[opt_bs_num])
        #print("opt_bs_num:"+str(opt_bs_num))

        
        ####debug only    
        #self.UE_list[0].set_xy([200, 190])
        #self.UE_list[1].set_xy([200, 210])
        #self.UE_list[2].set_xy([190, 210])
        ##self.UE_list[0].set_xy([20, 20])
        ##self.UE_list[1].set_xy([380, 380])
        ##self.UE_list[1].set_xy([20, 380])
        #self.BS_list[0].set_xy([0, 0])
        #self.BS_list[1].set_xy([400, 400])
        #self.BS_list[2].set_xy([0, 400])
        
        #for i in self.UE_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
        #for i in self.BS_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
        
    
        for i in self.UE_list:
            #i.reset_connecting_BS()
            #self.find_close_bs(i) #???
            i.renew_time_stamp()

        for i in self.BS_list:
            i.gen_random_data(self.time_step // 10)

        self.set_send_recieve()

        #for i in self.UE_list:
        #    i.recieve_set()
        #print(str(self.time_slot) + ": recieve set")

            #for j in self.BS_list:
            #    print("bs:"+str(j.get_name()))

        #    i.send_set()
        #print(str(self.time_slot) + ": send set")

        #print("Link list:"+str(self.Link_list))


        for j in range(10):
            self.time_slot = self.time_step % 10
            #print("time slot:"+str(time_slot))

            #print("time_slot"+str(self.time_slot))
            
            for i in self.BS_list:
                i.set_state(self.time_slot)
    
            self.capacity_calculation()        

            for i in self.UE_list:
                i.recieve(self.time_slot)
            #print(str(self.time_slot) + ": recieve")
            

            for i in self.UE_list:
                i.send(self.time_slot)
            #print(str(self.time_slot) + ": send")

            #visualization
            self.log()

                     

            for i in self.UE_list:
                i.clear_links_ul()
                i.clear_links_dl()
            #print(str(self.time_slot) + ": clear")
            
            #print("exec_Link_list:"+str(self.Link_list))

            #for i in self.Link_list:
            #    print("Link_list last:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name())+", "+str(i.is_active()))


            self.time_step = self.time_step + 1


        for i in self.BS_list:
            i.get_observation(self.time_step)
            #print("real_obs:"+str(real_obs))

            if i.get_type() == "SBS":
                ans1_1, ans1_2 = i.get_bs_throughput()
                #reward = (ans1_2 + ans1_2) / 2.0
                reward = ans1_2 * ans1_2
                print("reward:"+str(reward)+", "+str(i.get_name()))
                i.learning(reward)
            else:
                if self.time_step % i.mbs_interval == 0:
                    ans1_1, ans1_2 = i.get_bs_throughput()
                    #reward = (ans1_2 + ans1_2) / 2.0
                    reward = ans1_2 * ans1_2
                    print("reward:"+str(reward)+", "+str(i.get_name()))
                    i.learning(reward)

                
###################

    def execute_schemeD(self):


        #for i in self.BS_list:
            #i.initialize_observation()
            
        
        #opt_bs_num = (self.time_step // 10) % len(self.BS_list)
        #for i in self.BS_list:
        #    i.set_opt_bs(self.BS_list[opt_bs_num])
        #print("opt_bs_num:"+str(opt_bs_num))

        
        ####debug only    
        #self.UE_list[0].set_xy([200, 190])
        #self.UE_list[1].set_xy([200, 210])
        #self.UE_list[2].set_xy([190, 210])
        ##self.UE_list[0].set_xy([20, 20])
        ##self.UE_list[1].set_xy([380, 380])
        ##self.UE_list[1].set_xy([20, 380])
        #self.BS_list[0].set_xy([0, 0])
        #self.BS_list[1].set_xy([400, 400])
        #self.BS_list[2].set_xy([0, 400])
        
        #for i in self.UE_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
        #for i in self.BS_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
        
   
        for i in self.UE_list:
            #i.reset_connecting_BS()
            #self.find_close_bs(i) #???
            i.renew_time_stamp()
                
        for i in self.BS_list:
            i.gen_random_data(self.time_step // 10)

        self.set_send_recieve()

        #for i in self.UE_list:
        #    i.recieve_set()
        #print(str(self.time_slot) + ": recieve set")

            #for j in self.BS_list:
            #    print("bs:"+str(j.get_name()))

        #    i.send_set()
        #print(str(self.time_slot) + ": send set")

        #print("Link list:"+str(self.Link_list))

        for j in range(10):
            self.time_slot = self.time_step % 10
            #print("time slot:"+str(time_slot))
            
            
            #print("time_slot"+str(self.time_slot))

            
            for i in self.BS_list:
                i.set_state(self.time_slot)
    
            self.capacity_calculation()        

            #for i in self.Link_list:
            #    print("after capacity:"+str(i.get_bs().get_name())+", "+str(i.get_ue().get_name()))


            if self.time_slot < 5:
                for i in self.UE_list:
                    if i.is_ERUE == True:
                        i.recieve(self.time_slot)
                for i in self.UE_list:
                    if i.is_ERUE == False:
                        i.recieve(self.time_slot)
            else:
                for i in self.UE_list:
                    if i.is_ERUE == False:
                        i.recieve(self.time_slot)
                for i in self.UE_list:
                    if i.is_ERUE == True:
                        i.recieve(self.time_slot)
            #print(str(self.time_slot) + ": recieve")
            

            for i in self.UE_list:
                i.send(self.time_slot)
            #print(str(self.time_slot) + ": send")

            #visualization
            self.log()

                     
            for i in self.UE_list:
                i.clear_links_ul()
                i.clear_links_dl()
            #print(str(self.time_slot) + ": clear")
            
            #for i in self.Link_list:
            #    print("exec_Link_list:"+str(i.get_bs().get_name())+", "+str(i.get_ue().get_name()))


            #for i in self.BS_list:
            #    print("before BS:"+str(i.get_name()))

            ####
            for i in self.BS_list:
                if i.get_type() == "MBS":
                    ue = i.find_ue_week_SINR()
                    if ue != None:
                        sbs = ue.find_sbs_large_RSRP()
                        #d_mbs_dl = i.get_connecting_UE() / i.get_down_num()
                        #d_mbs_ul = i.get_connecting_UE() / i.get_up_num()
                        d_mbs_dl_next = (i.get_connecting_UE_num() - 1) / i.get_down_num()
                        d_mbs_ul_next = (i.get_connecting_UE_num() - 1) / i.get_up_num()
                        
                        if (sbs.get_dynamic_num() - sbs.get_up_num()) > 0:
                            d_sbs_dl = sbs.get_connecting_ERUE_num() / (sbs.get_dynamic_num() - sbs.get_up_num())
                        else:
                            d_sbs_dl = 0
                            
                        d_sbs_ul = (sbs.get_connecting_UE_num() + 1) / sbs.get_up_num()
                        
                        #print("d_sbs_dl:"+str(d_sbs_dl))
                        #print("d_mbs_dl_next:"+str(d_mbs_dl_next))

                        #print("d_sbs_ul:"+str(d_sbs_ul))
                        #print("d_mbs_ul_next:"+str(d_mbs_ul_next))

                        if (d_sbs_dl < d_mbs_dl_next) and (d_sbs_ul < d_mbs_ul_next):
                            #print("offload")
                            ue.set_connecting_BS(sbs)
                            ue.set_inside_BS(sbs)
                            ue.is_ERUE = True
                            #direction = i.remove_connecting_UE(ue)
                            i.remove_inside_UE(ue)
                            #if direction == "ul":
                            if ue in i.get_connecting_UE_ul():
                                i.remove_connecting_UE_ul(ue)
                                sbs.set_connecting_UE_ul(ue)
                            #else:
                            if ue in i.get_connecting_UE_dl():
                                i.remove_connecting_UE_dl(ue)
                                sbs.set_connecting_UE_dl(ue)
                            sbs.set_inside_UE(ue)
                            for env_t, (tbs, _, _, _, _, _) in enumerate(ue.data_list):
                                if tbs == i:
                                    ue.data_list[env_t][0] = sbs
                                                        
                            for j_env, j in enumerate(self.Link_list):
                                if (j.get_ue() == ue) and (j.get_bs() == i):
                                    self.Link_list[j_env].change_bs(sbs)
                                    #print("after change_bs:"+str(self.Link_list[j_env].get_bs().get_name()))

            #for i in self.BS_list:
            #    print("after BS:"+str(i.get_name()))
            #    for j in i.get_connecting_UE():
            #        print("ue:"+str(j.get_name()))

            for i in self.BS_list:
                if i.get_type() == "SBS":
                    min_tobj = 10000
                    min_j = 0
                    #print("down_num"+str(i.get_dynamic_num()))
                    for j in range(1,i.get_dynamic_num()):
                        #print("ERUE:"+str(i.get_connecting_ERUE_num()))
                        #print("UE:"+str(i.get_connecting_UE_num()))
                        d_sbs_dl = i.get_connecting_ERUE_num() / (i.get_dynamic_num() - j)
                        d_sbs_ul = (i.get_connecting_UE_num() + 1) / j
                        tobj = abs(d_sbs_dl - d_sbs_ul)
                        if tobj < min_tobj:
                            #print("tobj:"+str(tobj))
                            #print("j:"+str(j))
                            min_tobj = tobj
                            min_j = j
                    i.set_up_num(min_j)
                    print("config_num:"+str(6-min_j))
            ####



            self.time_step = self.time_step + 1


        ####for i in self.BS_list: !!!!!!!!!!!!!
        ####    real_obs = i.get_observation()
            #print(str(i.get_name())+" real_reward:"+str(real_reward))


####################


    def execute(self):


        #for i in self.BS_list:
            #i.initialize_observation()
            
        #opt_bs_num = (self.time_step // 10) % len(self.BS_list)
        #for i in self.BS_list:
        #    i.set_opt_bs(self.BS_list[opt_bs_num])
        #print("opt_bs_num:"+str(opt_bs_num))

        
        ####debug only    
        #self.UE_list[0].set_xy([200, 190])
        #self.UE_list[1].set_xy([200, 210])
        #self.UE_list[2].set_xy([190, 210])
        ##self.UE_list[0].set_xy([20, 20])
        ##self.UE_list[1].set_xy([380, 380])
        ##self.UE_list[1].set_xy([20, 380])
        #self.BS_list[0].set_xy([0, 0])
        #self.BS_list[1].set_xy([400, 400])
        #self.BS_list[2].set_xy([0, 400])
        
        #for i in self.UE_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
        #for i in self.BS_list:
        #    print(str(i.get_name())+" position:"+str(i.get_xy()))
           
        for i in self.UE_list:
#            i.reset_connecting_BS()
#            self.find_close_bs(i)
            i.renew_time_stamp()

        for i in self.BS_list:
            i.gen_random_data(self.time_step // 10)

        #print("AAA4")
        self.set_send_recieve()
            #i.recieve_set()
        #print(str(self.time_slot) + ": recieve set")
            #i.send_set()
        #print(str(self.time_slot) + ": send set")


        #for i in self.Link_list:
        #    print("Link list:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name()))


        for j in range(10):
            self.time_slot = self.time_step % 10
            print("time slot:"+str(self.time_slot))

            #print("time_slot"+str(self.time_slot))
            
            for i in self.BS_list:
                i.set_state(self.time_slot)
    
            self.capacity_calculation()        

            for i in self.UE_list:
                i.recieve(self.time_slot)
            #print(str(self.time_slot) + ": recieve")
            

            for i in self.UE_list:
                i.send(self.time_slot)
            #print(str(self.time_slot) + ": send")

            #visualization
            self.log()

                     

            for i in self.UE_list:
                i.clear_links_ul()
                i.clear_links_dl()
            #print(str(self.time_slot) + ": clear")
            
            #print("exec_Link_list:"+str(self.Link_list))

            #for i in self.Link_list:
            #    print("Link_list last:"+str(i.get_s().get_name())+", "+str(i.get_d().get_name())+", "+str(i.is_active()))


            self.time_step = self.time_step + 1


        for i in self.BS_list:
            real_obs = i.get_observation()
            #print(str(i.get_name())+" real_reward:"+str(real_reward))


#    def find_close_bs(self,ue):
#    
#        #print("BS_list:"+str(self.BS_list))
#        distance_SBS = []
#        distance_MBS = []
#        for i_env, i in enumerate(self.BS_list):
#            if i.get_type() == "SBS":
#                distance_SBS.append((i_env, calc_d(i.get_xy(), ue.get_xy())))
#            else:
#                distance_MBS.append((i_env, calc_d(i.get_xy(), ue.get_xy())))
#                
#        #print("distance:"+str(distance_SBS))
#        #print("distance:"+str(distance_MBS))
#
#        min_distance_SBS = 1000
#        min_i_env = 0
#        for i_env, i in distance_SBS:
#            if i < min_distance_SBS:
#                min_distance_SBS = i
#                min_i_env = i_env            
#
#        if min_distance_SBS > 40:
#            min_distance_MBS = 1000
#            min_i_env = 0
#            for i_env, i in distance_MBS:
#                if i < min_distance_MBS:
#                    min_distance_MBS = i
#                    min_i_env = i_env            
#
#        #print("min_i_env:"+str(min_i_env))
#        
#        #print("UE:"+str(ue.get_name())+" BS:"+str(self.BS_list[min_i_env].get_name()) + " Distance:" +str(calc_d(self.BS_list[min_i_env].get_xy(), ue.get_xy())))
#
#        ans_BS = self.BS_list[min_i_env]
#
#        return(ans_BS)
        
    def visualization(self):
    
        G = nx.Graph()
        
        for i in self.UE_list:
            G.add_node(i.get_name(), pos=i.get_generalized_xy())
            print(i.get_xy())
        for i in self.BS_list:
            print("BS:"+str(i.get_name())+", config id:"+str(i.get_config_id()))
            G.add_node(i.get_name(), pos=i.get_generalized_xy())
            print(i.get_xy())

#        for i in self.DL_list:
#            print("BS:"+str(i.get_bs().get_name)+", state:"+str(i.get_bs().get_state()))
#            if i.get_bs().get_state() == "D":
#                G.add_edge(i.get_bs().get_name(), i.get_ue().get_name(), color='r')
#            elif  i.get_bs().get_state() == "U":
#                G.add_edge(i.get_bs().get_name(), i.get_ue().get_name(), color='b')            

#        for i in self.UL_list:
#            print("BS:"+str(i.get_bs().get_name)+", state:"+str(i.get_bs().get_state()))
#            if i.get_bs().get_state() == "D":
#                G.add_edge(i.get_ue().get_name(), i.get_bs().get_name(), color='r')
#            elif i.get_bs().get_state() == "U":
#                G.add_edge(i.get_ue().get_name(), i.get_bs().get_name(), color='b')
            
        
        #print("nodes:"+str(nx.nodes(G)))
        #print("edges:"+str(nx.edges(G)))
        
        
        colors = nx.get_edge_attributes(G,'color').values()
        
        nx.draw(G, nx.get_node_attributes(G, 'pos'), edge_color=colors, with_labels=True, node_size=0)
        #plt.pause(.001)

