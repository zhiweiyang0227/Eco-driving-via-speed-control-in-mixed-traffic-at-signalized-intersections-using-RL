# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 21:52:51 2022

@author: s4680600
"""

CA = True
total_episode = 3000
TTC_threshold = 4.001
base_name = f'yzw_{TTC_threshold}_CA' 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow import keras
import random
import numpy as np
from env_HybridByRules import Env
import scipy.io as sio
import pickle as pk
import sys
import os
import matplotlib.pyplot as plt
import math
from sympy import *
os.environ["CUDA_VISIBLE_DEVICES"] = ""

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)

with tf.compat.v1.Session(config=config) as sess:
    #... this will run single threaded
    pass
tf.compat.v1.reset_default_graph()
# if os.environ.get("PYTHONHASHSEED") != "0":
#     raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")
# os.environ['PYTHONHASHSEED'] = '0'
random_no = 3  #0only v larger 1all bad 2all better 3only fuel better
random.seed(random_no)
np.random.seed(random_no)
# tf.compat.v1.set_random_seed(2)  # reproducible
tf.random.set_seed(random_no)
base_name = f'yzw_{TTC_threshold}_CA_'+str(random_no) 

# LR 0.001 0.0015 bs 512 safety 0.9 not that converge
#####################  hyper parameters  ####################
LR_A = 0.001  # learning rate for actor 0.0005 not that converge efficiency is worse 0.0001 converge safe worse efficiency worse
LR_C = 0.0015  # learning rate for critic 0.0015 safety is worse but others are better 0.002 not converge0.0017safety too bad fuel similar
GAMMA = 0.99   # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300 #hard replacement
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 150000  # 7000
BATCH_SIZE = 256#512 x

###############################  Actor  ####################################
#tf.compat.v1.reset_default_graph()
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        
        self.sess = tf.compat.v1.Session()
        #self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.compat.v1.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], 'r')

        with tf.compat.v1.variable_scope('Actor'):#, reuse=tf.compat.v1.AUTO_REUSE):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.compat.v1.variable_scope('Critic'):#, reuse=tf.compat.v1.AUTO_REUSE):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        
        # target net replacement
        self.soft_replace = [tf.compat.v1.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]
        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.compat.v1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.compat.v1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q  
        self.atrain = tf.compat.v1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, replace):
        # if self.a_replace_counter % REPLACE_ITER_A == 0 or replace:
        #     self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
            
        # if self.c_replace_counter % REPLACE_ITER_C == 0 or replace:
        #     self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
            
        # self.a_replace_counter += 1;
        # self.c_replace_counter += 1

        # if self.pointer < MEMORY_CAPACITY:
        #     indices = np.random.choice(self.pointer, size=BATCH_SIZE)
        # else:
        #     indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})  # {}place holder 处理输入

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            net = keras.layers.Dense(30, activation=tf.nn.relu, name='l1', trainable=trainable)(s)
            # net = keras.layers.Dense(128, activation=tf.nn.relu, name='l2', trainable=trainable)(net)
            # net = keras.layers.Dense(128, activation=tf.nn.relu, name='l3', trainable=trainable)(net)
            a = keras.layers.Dense(self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)(net)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            n_l1 = 30
            n_l2 = 128
            n_l3 = 128
            # hidden_1 = keras.layers.Dense(n_l1, activation=tf.nn.elu, name='hidden_1')(s)
            # hidden_2 = keras.layers.Dense(n_l2, activation=tf.nn.elu, name='hidden_2')(hidden_1)
            
            w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            # w1_s = tf.compat.v1.get_variable('w2_s', [self.s_dim, n_l2], trainable=trainable)
            # w1_s = tf.compat.v1.get_variable('w3_s', [self.s_dim, n_l3], trainable=trainable)
            
            w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            # w1_a = tf.compat.v1.get_variable('w2_a', [self.a_dim, n_l2], trainable=trainable)
            # w1_a = tf.compat.v1.get_variable('w3_a', [self.a_dim, n_l3], trainable=trainable)            
            
            b1 = tf.compat.v1.get_variable('b1', [1, n_l1], trainable=trainable)
            # b1 = tf.compat.v1.get_variable('b2', [1, n_l2], trainable=trainable)
            # b1 = tf.compat.v1.get_variable('b3', [1, n_l3], trainable=trainable)
            
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # net = keras.layers.Dense(n_l2, trainable=trainable)(net)
            # net = keras.layers.Dense(n_l3, trainable=trainable)(net)
            return keras.layers.Dense(1, trainable=trainable)(net)  # Q(s,a)

    def savenet(self, file):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, 'save/' + file + '.ckpt')

    def restore(self, file):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, 'save/' + file + '.ckpt')

env = Env(TTC_threshold)
s_dim = env.n_features
a_dim = env.n_actions
a_bound = env.action_Bound

# load training data
train = sio.loadmat('trainSet.mat')['calibrationData']
test = sio.loadmat('testSet.mat')['validationData']
trainNum = train.shape[0]
testNum = test.shape[0]
print('Number of training samples:', trainNum)
print('Number of validate samples:', testNum)

# # Stop distance collision avoidance
rolling_window = 100  # 100 car following events, average score
result = []


for run in [base_name]:
    # name is the name of the experiment, CA is whether use collision avoidance
    ddpg = DDPG(a_dim, s_dim, a_bound)

    # training part
    max_rolling_score = np.float('-inf')
    max_score = np.float('-inf')
    var = 3
    collision_train = 0
    TL_violation_train =0
    episode_score = np.zeros(total_episode)  # average score of each car following event
    rolling_score = np.zeros(total_episode)
    cum_collision_num = np.zeros(total_episode)
    cum_TL_violation_num = np.zeros(total_episode)
    score_safe = np.zeros(total_episode)
    score_efficiency = np.zeros(total_episode)
    score_comfort = np.zeros(total_episode)
    score_action = np.zeros(total_episode)
    score_fuel = np.zeros(total_episode)
    score_SPaT = np.zeros(total_episode)
    score_collision = np.zeros(total_episode)
    score_IDMCF = np.zeros(total_episode)
    for i in range(total_episode):
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
            # Add exploration noise
            a = ddpg.choose_action(s)[0]
            a = np.clip(np.random.normal(a, var),max(-a_bound, a_bound_v[0]), min(a_bound,a_bound_v[1]))  # add randomness to action selection for exploration
            if CA:
                state = env.s
                space, svSpd, relSpd, d_tl, g_s, g_e = state
                lvSpd = svSpd - relSpd
                RT = 0.5  # reaction time
                s_s = 1#.5
                critical_hdw_ttc = 1.5
                s_stop = 0.5
                a_TL = None
                a_col = None
                #SD = svSpd * RT + (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound)

                if env.SimPosData[env.timeStep-1]<=470:
                    estimated_arr = data[env.timeStep,5]+(470-env.SimPosData[env.timeStep-1]-svSpd*0.04-0.5*a*0.04**2)/(svSpd+a*0.04)
                    estimated_arr_TL,_ = env.TL(estimated_arr)
                    if estimated_arr_TL is 'red':
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
                    a3 = (space-relSpd*0.04-svSpd * RT - (svSpd ** 2 - lvSpd ** 2) / (2 * a_bound))/(0.5*0.04**2)
                    a_col = max(-a_bound,min(a1,a2,a3))
                if a_TL is not None and a_col is not None:
                    a = max(-a_bound,min(a_TL, a_col))
                elif a_TL is None and a_col is not None:
                    a = max(-a_bound,a_col)
                elif a_TL is not None and a_col is None:
                    a = max(-a_bound,a_TL)
            action_list.append(a)

            #print('dis', dis)
            s_, r, done, r_info, a_bound_jerk, a_bound_v = env.step(a, total_fuel, dis,data,TTC_violate, fTTC_total)
            #             sys.stdout.write(f'\r TTC: {r_info[0]}, hdw: {r_info[1]}, jerk: {r_info[2]}, fTTC: {r_info[3]}, fHdw: {r_info[4]}, fJerk: {r_info[5]}')
            #             sys.stdout.flush()

            #print('r_info', r_info)
            ddpg.store_transition(s, a, r, s_)
            replace = False
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn(replace)

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
                    score = score_f +score_tl+score_col+ score_c+ score_a + score_e+score_IDM#score_s + 
                    # score = -1
                    collision_train += 1
                else:
                    score = score_s + score_f +score_tl+ score_c+ score_a + score_e+score_IDM#+score_col
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
        
        if max_score < score:
            max_score = score

        if rolling_score[i] > max_rolling_score:
            max_rolling_score = rolling_score[i]
            # save network parameters
            ddpg.savenet(f'model_{run}_'+str(i))

        sys.stdout.write(
            f'''\r Run {run}, Episode {i}, Score: {score:.2f}, Rolling score: {rolling_score[i]:.2f}, Max score: {max_score:.2f}, Max rolling score: {max_rolling_score:.2f}, collisions: {collision_train}, TL violations: {TL_violation_train}, score_safe: {score_s:.2f}, score_efficiency: {score_e:.2f}, score_comfort: {score_c:.2f}, score_action: {score_a:.2f}, score_fuel: {score_f:.2f}, score_SPaT: {score_tl:.2f}, score_col: {score_col:.2f}, score_IDM: {score_IDM:.2f}   ''')
        if score_col==-env.penalty:
            print('space', env.space)
        elif score_tl <=0 and env.SimPosData[env.timeStep-1] >470:
            print('score_tl',score_tl)
        else:
            sys.stdout.flush()
            

        # save results
        # result.append([episode_score, rolling_score, cum_collision_num, cum_TL_violation_num, score_safe, score_efficiency, score_comfort, score_action,score_fuel,score_SPaT,score_collision])
        if (i+1)%100==0:
            # np.save(f'result_{run}'+str(i+1)+'.npy', result)
            # ddpg.savenet(f'model_{run}'+str(i+1))
            plt.plot(rolling_score)
            plt.show()
score_summary = pd.DataFrame(np.vstack([episode_score, rolling_score, score_fuel, score_safe, score_efficiency, score_comfort, score_action, score_SPaT, score_collision, score_IDMCF]).T)
score_summary.columns=['episode_score', 'rolling_score', 'score_fuel', 'score_safe', 'score_efficiency', 'score_comfort', 'score_action', 'score_SPaT', 'score_collision', 'score_IDM']
writer = pd.ExcelWriter('save/RL_bestTrain/score_data_training_{}.xlsx'.format(random_no))		# 写入Excel文件
score_summary.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.save()
writer.close()
            