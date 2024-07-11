# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:21:39 2022

@author: s4680600
"""

CA = True
EPISODE = 1000
TTC_threshold = 4.001
base_name = f'yzw_{TTC_threshold}_CA' 

random_no = 3

import numpy as np
# np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import random
import time
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal


import GPUtil
import psutil
from threading import Thread
import time

from simulation_env import Env

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from collections import namedtuple
import collections, random

import scipy.io as sio
import pandas as pd
import numpy as np

torch.manual_seed(random_no)
random.seed(random_no)
np.random.seed(random_no)
base_name = f'yzw_{TTC_threshold}_CA_'+str(random_no) 

class ReplayBuffer():
    def __init__(self, buffer_limit, DEVICE):
        self.buffer = deque(maxlen=buffer_limit)
        self.dev = DEVICE
 
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
        a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
        r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
        done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

        # r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, DEVICE):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)# latter 64
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_std = nn.Linear(128, action_dim)

        self.lr = actor_lr
        self.dev = DEVICE

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = torch.FloatTensor([4]).to(self.dev)
        self.min_action = torch.FloatTensor([-4]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        
        self.to(self.dev)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob
        
        
        

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, critic_lr, DEVICE):
        super(QNetwork, self).__init__()

        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim, 64)
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 128)#latter 64
        self.fc_out = nn.Linear(128, action_dim)

        self.lr = critic_lr
        self.dev = DEVICE
        
        self.to(self.dev)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, a):
        h1 = F.leaky_relu(self.fc_s(x))
        h2 = F.leaky_relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=-1)
        q = F.leaky_relu(self.fc_1(cat))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q

class SAC_Agent:
    def __init__(self):
        self.state_dim      = env.n_features
        self.action_dim     = env.n_actions 
        self.lr_pi          = 0.001#0.001#0.0001
        self.lr_q           = 0.002#0.0015#0.0005
        self.gamma          = 0.99#0.98#0.999
        self.batch_size     = 512#200
        self.buffer_limit   = 150000#300000
        self.tau            = 0.001#0.0007   # for soft-update of Q using Q-target
        self.init_alpha     = 8
        self.target_entropy = -2 #-self.action_dim  # == -2
        self.lr_alpha       = 0.001
        self.DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory         = ReplayBuffer(self.buffer_limit, self.DEVICE)
        print("Device used:", self.DEVICE)

        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

        self.PI  = PolicyNetwork(self.state_dim, self.action_dim, self.lr_pi, self.DEVICE)
        self.Q1        = QNetwork(self.state_dim, self.action_dim, self.lr_q, self.DEVICE)
        self.Q1_target = QNetwork(self.state_dim, self.action_dim, self.lr_q, self.DEVICE)
        self.Q2        = QNetwork(self.state_dim, self.action_dim, self.lr_q, self.DEVICE)
        self.Q2_target = QNetwork(self.state_dim, self.action_dim, self.lr_q, self.DEVICE)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def choose_action(self, s):
        with torch.no_grad():
            action, log_prob = self.PI.sample(s.to(self.DEVICE))
        return action, log_prob

    def calc_target(self, mini_batch):
        s, a, r, s_prime, done = mini_batch
        with torch.no_grad():
            a_prime, log_prob_prime = self.PI.sample(s_prime)
            entropy = - self.log_alpha.exp() * log_prob_prime
            q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
            q_target = torch.min(q1_target, q2_target)
            target = r + self.gamma * done * (q_target + entropy)
        return target

    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch

        td_target = self.calc_target(mini_batch)

        #### Q1 train ####
        q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
        self.Q1.optimizer.zero_grad()
        q1_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q1.parameters(), 1.0)
        self.Q1.optimizer.step()
        #### Q1 train ####

        #### Q2 train ####
        q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
        self.Q2.optimizer.zero_grad()
        q2_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.Q2.parameters(), 1.0)
        self.Q2.optimizer.step()
        #### Q2 train ####

        #### pi train ####
        a, log_prob = self.PI.sample(s_batch)
        entropy = -self.log_alpha.exp() * log_prob

        q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
        q = torch.min(q1, q2)

        pi_loss = -(q + entropy)  # for gradient ascent
        self.PI.optimizer.zero_grad()
        pi_loss.mean().backward()
        nn.utils.clip_grad_norm_(self.PI.parameters(), 1.0)
        self.PI.optimizer.step()
        #### pi train ####

        #### alpha train ####
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        #### alpha train ####

        #### Q1, Q2 soft-update ####
        for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        #### Q1, Q2 soft-update ####
        



class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization()
            print("CPU percentage: ", psutil.cpu_percent())
            print('CPU virtual_memory used:', psutil.virtual_memory()[2], "\n")
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
        

# GPU_CPU_monitor = Monitor(60)

writer = SummaryWriter('SAC_log')
env = Env(TTC_threshold)
agent = SAC_Agent()

a_bound = env.action_Bound

# load training data
train = sio.loadmat('trainSet.mat')['calibrationData']
test = sio.loadmat('testSet.mat')['validationData']
trainNum = train.shape[0]
testNum = test.shape[0]
print('Number of training samples:', trainNum)
print('Number of validate samples:', testNum)
rolling_window = 100



for run in [base_name]:
    # training part
    max_rolling_score = np.float64('-inf')
    max_score = np.float64('-inf')
    collision_train = 0
    TL_violation_train =0
    episode_score = np.zeros(EPISODE)  # average score of each car following event
    rolling_score = np.zeros(EPISODE)
    cum_collision_num = np.zeros(EPISODE)
    cum_TL_violation_num = np.zeros(EPISODE)
    score_safe = np.zeros(EPISODE)
    score_efficiency = np.zeros(EPISODE)
    score_comfort = np.zeros(EPISODE)
    score_action = np.zeros(EPISODE)
    score_fuel = np.zeros(EPISODE)
    score_SPaT = np.zeros(EPISODE)
    score_collision = np.zeros(EPISODE)
    score_IDMCF = np.zeros(EPISODE)
    for i in range(EPISODE): 
        car_fol_id = np.random.randint(0, trainNum - 1)
        data = train[car_fol_id, 0]
        s, last_action = env.reset(data)
        first_step_duration = data[1,5]-data[0,5]
        a_bound_jerk=[last_action-first_step_duration*10,last_action+first_step_duration*10]
        a_bound_v=[-s[1]/first_step_duration,(50/3.6-s[1])/first_step_duration]
        score = 0
        score_s, score_e, score_c, score_a, score_tl, score_col, score_IDM = 0, 0, 0, 0, 0, 0, 0  # part objective scores
        ori_loc = data[0,7]
        dis = 0
        total_fuel = 0
        fTTC_total =0
        TTC_violate =0
        action_list = [last_action]        
        while True:
            a, log_prob = agent.choose_action(torch.FloatTensor(s))
            a = float(a.detach().cpu().numpy())
            a = np.clip(a,max(-a_bound, a_bound_v[0]), min(a_bound,a_bound_v[1]))
            if CA:
                state = env.s
                space, svSpd, relSpd, d_tl, g_s, g_e = state
                lvSpd = svSpd - relSpd
                RT = 0.5  # reaction time
                s_s = 1.5#.5
                critical_hdw_ttc = 1.5
                s_stop = 0.5
                a_TL = None
                a_col = None
                #SD = svSpd * RT + (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound)
            
                if env.SimPosData[env.timeStep-1]<=470:
                    estimated_arr = data[env.timeStep,5]+(470-env.SimPosData[env.timeStep-1]-svSpd*0.04-0.5*a*0.04**2)/(svSpd+a*0.04)
                    estimated_arr_TL,_ = env.TL(estimated_arr)
                    if estimated_arr_TL == 'red':
                        SD = s_stop+(svSpd ** 2) / (2 * a_bound)
                        if 470-env.SimPosData[env.timeStep-1]-svSpd*0.04-0.5*a*0.04**2< SD:
                            a_TL = -a_bound
            
                # add collision avoidance guidance
                if relSpd>0:
                    SD = max(critical_hdw_ttc*(relSpd+a*0.04)+s_s, s_s+critical_hdw_ttc*(svSpd+a*0.04)-(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5,svSpd * RT + (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound))
                else:
                    SD = max(s_s+critical_hdw_ttc*(svSpd+a*0.04)-(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5,svSpd * RT + (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound))
                if space-relSpd*0.04-0.5*a*0.04**2< SD:
                    # x = Symbol('x')
                    # a1= solve(critical_hdw_ttc*(relSpd+x*0.04)+s_s-(space-relSpd*0.04-0.5*x*0.04**2),x)
                    # critical_hdw_ttc*(relSpd+x*0.04)+s_s=space-relSpd*0.04-0.5*x*0.04**2
                    # critical_hdw_ttc*x*0.04+0.5*x*0.04**2=space-relSpd*0.04-critical_hdw_ttc*relSpd-s_s
                    a1 = (space-relSpd*0.04-critical_hdw_ttc*relSpd-s_s)/(critical_hdw_ttc*0.04+0.5*0.04**2)
                    # a2=solve(s_s+critical_hdw_ttc*(svSpd+x*0.04)-(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5-(space-relSpd*0.04-0.5*x*0.04**2),x)
                    # s_s+critical_hdw_ttc*(svSpd+x*0.04)-(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5=space-relSpd*0.04-0.5*x*0.04**2
                    # critical_hdw_ttc*x*0.04+0.5*x*0.04**2=space-relSpd*0.04-critical_hdw_ttc*svSpd-s_s+(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5
                    a2 = (space-relSpd*0.04-critical_hdw_ttc*svSpd-s_s+(data[env.timeStep,8]+data[env.timeStep-1,11])*0.5)/(critical_hdw_ttc*0.04+0.5*0.04**2)
                    # svSpd * RT + (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound) = space-relSpd*0.04-0.5*a*0.04**2
                    # 0.5*a*0.04**2=space-relSpd*0.04-svSpd * RT - (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound)
                    a3 = (space-relSpd*0.04-svSpd * RT  - (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound))/(0.5*0.04**2)
                    a_col = max(-a_bound,min(a1,a2,a3))
                if a_TL is not None and a_col is not None:
                    a = max(-a_bound,min(a_TL, a_col))
                elif a_TL is None and a_col is not None:
                    a = max(-a_bound,a_col)
                elif a_TL is not None and a_col is None:
                    a = max(-a_bound,a_TL)
            action_list.append(a)
            s_, r, done, r_info, a_bound_jerk, a_bound_v = env.step(a, total_fuel, dis,data,TTC_violate, fTTC_total)
            agent.memory.put((s, a, r, s_, done))
            if agent.memory.size() > agent.batch_size:
                agent.train_agent()
            
            s = s_
            #score += r
            #score_s += r_info[3]
            #score_e += r_info[4]
            score_c += r_info[5]
            score_a += r_info[6]
            score_IDM += r_info[14]
            #score_col+=r_info[11]
            total_fuel = r_info[7]
            dis = r_info[10]#(20**(math.pow(r_info[8],1/5))-1)*1000*r_info[7]
            TTC_violate = r_info[12]
            fTTC_total = r_info[13]
            

            
            
            if done:
                duration = data.shape[0]
                #score /= duration  # normalize with respect to car-following length
                #score_s /= duration
                #score_col /= duration
                #score_e /= duration
                score_c /= duration
                score_a /= duration
                score_IDM /= duration
                if TTC_violate !=0:
                    score_s= fTTC_total/TTC_violate
                else:
                    score_s = 0                
                # score_s = fTTC_total/TTC_violate#r_info[3]
                score_f = r_info[8]
                score_e = r_info[4]
                score_tl = r_info[9]
                score_col =r_info[11]
                score_safe[i] = score_s
                score_fuel[i] = score_f
                score_efficiency[i] = score_e
                score_SPaT[i] = score_tl
                score_collision[i]=score_col
                
                if score_col==-env.penalty:#env.isCollision == 1:
                    score =score_c+ score_a +score_f +score_tl+score_e+score_col+ score_IDM#score_s +
                    # score = -1
                    collision_train += 1
                else:
                    score =score_c+ score_a +score_f +score_tl+score_e+score_s + score_IDM#+score_col
                if score_tl <=0 and env.SimPosData[env.timeStep-1] >470:
                    TL_violation_train+=1
                break


        # record episode results
        episode_score[i] = score
        #score_safe[i] = score_s
        #score_efficiency[i] = score_e
        score_comfort[i] = score_c
        score_action[i] = score_a
        score_IDMCF[i]=score_IDM
        #score_fuel[i] = score_f
        #score_collision[i]=score_col
        rolling_score[i] = np.mean(episode_score[max(0, i - rolling_window + 1):i + 1])
        cum_collision_num[i] = collision_train
        cum_TL_violation_num[i]=TL_violation_train
        
        writer.add_scalar("Score", score)    
        
        if max_score < score:
            max_score = score

        if rolling_score[i] > max_rolling_score:
            max_rolling_score = rolling_score[i]
            # save network parameters
            torch.save(agent.PI.state_dict(), "save/sac_actor_"+str(i)+".pt")


        sys.stdout.write(
            f'''\r Run {run}, Episode {i}, Score: {score:.2f}, Rolling score: {rolling_score[i]:.2f}, Max score: {max_score:.2f}, Max rolling score: {max_rolling_score:.2f}, collisions: {collision_train}, TL violations: {TL_violation_train}, score_safe: {score_s:.2f}, score_efficiency: {score_e:.2f}, score_comfort: {score_c:.2f}, score_action: {score_a:.2f}, score_fuel: {score_f:.2f}, score_SPaT: {score_tl:.2f}, score_col: {score_col:.2f}, score_IDM: {score_IDM:.2f}   ''')
        if score_col==-env.penalty:
            print('space', env.space)
        elif score_tl <=0 and env.SimPosData[env.timeStep-1] >470:
            print('score_tl',score_tl)
        else:
            sys.stdout.flush()
            

        if (i+1)%100==0:
            # np.save(f'result_{run}'+str(i+1)+'.npy', result)
            # ddpg.savenet(f'model_{run}'+str(i+1))
            plt.plot(rolling_score)
            plt.show()
score_summary = pd.DataFrame(np.vstack([episode_score, rolling_score, score_fuel, score_safe, score_efficiency, score_comfort, score_action, score_SPaT, score_collision, score_IDMCF]).T)
score_summary.columns=['episode_score', 'rolling_score', 'score_fuel', 'score_safe', 'score_efficiency', 'score_comfort', 'score_action', 'score_SPaT', 'score_collision', 'score_IDM']
writer = pd.ExcelWriter('save/RL_bestTrain/score_data_training{}.xlsx'.format(random_no))		# 写入Excel文件
score_summary.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
