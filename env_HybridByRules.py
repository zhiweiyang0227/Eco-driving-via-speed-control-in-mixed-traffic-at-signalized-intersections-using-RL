# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:11:39 2022

@author: s4680600
"""


import math
import numpy as np
import copy
import pandas as pd
IVT = 2.5 #2.5
s_s = 1.5 #1.5
IVT_TL = 0.5
s_TL = 0.5
v_des = 50/3.6
v_max = 50/3.6
class Env(object):
    def __init__(self, TTC_threshold):
        self.action_Bound = 4
        self.n_actions = 1
        self.timeWindow = 1 # if you want to consider informaiton 
        #from previous seconds, you can use timewindow > 1
        self.penalty = 5 # penalty for collisions
        self.n_features = 6*self.timeWindow
        self.TTC_threshold = TTC_threshold



    def reset(self, data):
        temp = data[:self.timeWindow, [0,1,2]]
        _, fut_TL = self.TL(data[0,5])
        
        v_limit_TL = []
        for i in range(4):
            if fut_TL[i]-data[0,5]!=0:
                v_limit_TL.append(((470-data[0,7]))/(fut_TL[i]-data[0,5]))
            else:
                v_limit_TL.append(50/3.6)
        if pd.Interval(v_limit_TL[1],v_limit_TL[0]).overlaps(pd.Interval(0,50/3.6)):
            fut_g = [fut_TL[0],fut_TL[1]]
        elif pd.Interval(v_limit_TL[3],v_limit_TL[2]).overlaps(pd.Interval(0,50/3.6)):
            fut_g = [fut_TL[2],fut_TL[3]]
        else:
            fut_g = [fut_TL[0],fut_TL[1]]

        temp = np.append(temp,[[470-data[0,7], fut_g[0]-data[0,5],fut_g[1]-data[0,5]]],axis = 1).reshape(1, self.timeWindow * 6)
        temp = temp[0, :]
        self.s = temp
        self.currentState = self.s[-6:]
        relSpd = self.currentState[2]
        space = self.currentState[0]
        self.TTC = space / relSpd          
        self.isCollision = 0
        self.TimeLen = data.shape[0]
        self.lastAction = data[0,6]
        self.cut_in_time = None
        self.timeStep = self.timeWindow # starting form 1 to n
        self.LVSpdData = data[:, 3]
        self.LVAccData = np.zeros(data[:, 0].shape)
        self.SimSpaceData = np.zeros(data[:, 0].shape)
        self.SimSpeedData = np.zeros(data[:,1].shape)
        self.SimPosData = np.zeros(data[:, 0].shape)
        self.SimTimeData = np.zeros(data[:, 0].shape)
        self.SimScoreTLData = np.zeros(data[:, 0].shape)
        self.SimJerkData = np.zeros(data[:, 0].shape)
        self.SimTTCData = np.zeros(data[:, 0].shape)
        self.SimGreenStartData = np.zeros(data[:, 0].shape)
        self.SimGreenEndData = np.zeros(data[:, 0].shape)
        
        self.LVAccData[0] = 0
        self.SimSpaceData[0] = data[0,0] # initialize with initial spacing
        self.SimSpeedData[0] = data[0,1] # initialize with initial speed
        self.SimPosData[0] = data[0,7] # initialize with initial position
        self.SimTimeData[0] = data[0,5] # initialize with initial time
        self.SimScoreTLData[0] = 0 
        self.SimJerkData[0] = 0 
        self.SimTTCData[0] = self.TTC
        self.SimGreenStartData[0] = fut_g[0]
        self.SimGreenEndData[0] = fut_g[1]
        return self.s, self.lastAction
    
    def TL(self, time): # get TL color in predicting IDM trajectory

        cycle = 90

        phase_time = [16, 19, 46]

        if phase_time[0] <= (time+cycle) % cycle < phase_time[1]:
            TLcolor = "yellow"
            g = time
            r = time + (phase_time[1] - (time+cycle) % cycle)
            g_next = time + (phase_time[2] - (time+cycle) % cycle)
        elif phase_time[1] <= (time+cycle) % cycle < phase_time[2]:
            TLcolor = "red"
            g = time + (phase_time[2] - (time+cycle) % cycle)
            r = time - ((time+cycle) % cycle-phase_time[1])+cycle
            g_next = time + (phase_time[2] - (time+cycle) % cycle) + cycle
        else:
            TLcolor = "green"
            g = time
            if (time+cycle) % cycle<phase_time[0]:
                r=time + (phase_time[1] - (time+cycle) % cycle)
                g_next = time - (time+cycle) % cycle + cycle
            else:
                r = time - ((time+cycle) % cycle-phase_time[1])+cycle
                g_next = time - ((time+cycle) % cycle - phase_time[2]) + cycle
            
        return TLcolor, [g,r,g_next,g_next+cycle-phase_time[2]+phase_time[0]]
    
    def step(self, action, total_fuel, dis, data, TTC_violate, fTTC_total, del_a):
        def fuel_model(v, a):
            R = 1.2256/25.92*0.3*(0.85*2.015*1.748)*(v*3.6)**2+2000*9.8067*1.75/1000*(0.0328*v*3.6+4.575)+2000*9.8067*0
            P = max(0,(R+1.04*2000*a)/3600/0.9*v*3.6)
            fuel_r = 0.000341+0.0000583*P+0.000001*P*P
            return fuel_r
        def IDM(spacing, speed, delta_speed, timestep, IVT, s_s):

            a_bound = self.action_Bound

            s_des = s_s+max(speed*IVT+speed*delta_speed/2/math.sqrt(a_bound*a_bound),0)
            acce = max(-a_bound, min(a_bound, a_bound*(1-(speed/v_des)**4-(s_des/spacing)**2)))
            # print('acce', acce)
            # if acce >= acc_pre:
            #     acce = min(acc_pre + jerk_max*timestep, acce)
            # else:
            #     acce = max(acc_pre - jerk_max*timestep, acce)
            IDM_speed = max(0, min(v_max, (speed+acce*timestep)))
            acc = (IDM_speed-speed)/timestep

            return acc, IDM_speed
        # update state
        self.timeStep += 1
        time = data[self.timeStep-1,5]
        space_pre = data[self.timeStep-2, 7]+data[self.timeStep-2, 0]-data[0,7]-dis#self.currentState[0]-relSpd*step_duration
        speed_pre = self.currentState[1]
        LVSpd_pre = self.LVSpdData[self.timeStep-2]
        delta_spd = speed_pre-LVSpd_pre
        
        step_duration = (data[self.timeStep-1,5]-data[self.timeStep-2,5])
        LVSpd = self.LVSpdData[self.timeStep-1]
        svSpd = self.currentState[1] + action*step_duration
        fuel_rate = fuel_model(self.currentState[1], action)
        total_fuel += fuel_rate*step_duration
        dis = dis + self.currentState[1]*step_duration + 0.5*action*step_duration**2
        jerk = (action - self.lastAction) / step_duration
        
        #print('LVSpd, svSpd, spacing, fuel_rate, total_fuel, dis', LVSpd, svSpd, self.currentState[0], fuel_rate, total_fuel, dis)
       
        # if svSpd <= 0:
        #     svSpd = 0.00001
        #     self.isStall = 1
        # else:
        #     self.isStall = 0

        relSpd = svSpd-LVSpd
        space = data[self.timeStep-1, 7]+data[self.timeStep-1, 0]-data[0,7]-dis#self.currentState[0]-relSpd*step_duration
        self.space = space
        self.avgVehLen = (data[self.timeStep-2,8]+data[self.timeStep-2,11])*0.5
        hdw = (space + self.avgVehLen)/ svSpd
        self.TTC = space / relSpd 
        # judge cut in
        if data[self.timeStep-1,7]+data[self.timeStep-1,0]<data[self.timeStep-2,7]+data[self.timeStep-2,0]:
            self.cut_in_time = time
        #judge collision and back
        if space < 0:
            self.isCollision = 1
            if self.cut_in_time is not None:
                if time -self.cut_in_time<3.5:
                    fCol = 0
                else:
                    fCol = - self.penalty * self.isCollision
            else:
                fCol = - self.penalty * self.isCollision
        else:
            fCol = 0
            
        #store the space history for error calculating
        self.LVAccData[self.timeStep-1] = (LVSpd-LVSpd_pre)/step_duration
        self.SimSpaceData[self.timeStep-1] = space
        self.SimSpeedData[self.timeStep-1] = svSpd
        self.SimPosData[self.timeStep-1] = dis+data[0,7]
        self.SimTimeData[self.timeStep-1] = time
        self.SimJerkData[self.timeStep-1] = jerk
        self.SimTTCData[self.timeStep-1] = self.TTC
        
        # caculate the reward
        # TL_color = TL(time + (470-data[0,7]-dis)/svSpd)
        # if TL_color == "green" or TL_color == 'yellow':
        #     fTL = 1
        # else:
        #     fTL = -1
        

        v_limit_TL = []
        if dis+data[0,7]<=470:
            curr_TL, fut_TL = self.TL(time)
            d_tl = 470-data[0,7]-dis
            for i in range(4):
                if fut_TL[i]-time!=0:
                    v_limit_TL.append(min(50/3.6,((470-data[0,7]-dis))/(fut_TL[i]-time)))
                else:
                    v_limit_TL.append(50/3.6)
            if pd.Interval(float(v_limit_TL[1]),float(v_limit_TL[0])).overlaps(pd.Interval(0,50/3.6)):
                fut_g = [fut_TL[0],fut_TL[1]]
                v_limit = [max(v_limit_TL[1],0),min(v_limit_TL[0],50/3.6)]
                if svSpd >=v_limit[0] and svSpd<=v_limit[1]:
                    fTL = (1.5-0.5)/(v_limit[1]-v_limit[0])*svSpd+0.5-v_limit[0]/(v_limit[1]-v_limit[0])
                    # fTL = (1-0.5)/(v_limit[1]-v_limit[0])*svSpd+0.5-v_limit[0]/(v_limit[1]-v_limit[0])
                else:
                    fTL = -1
            elif pd.Interval(float(v_limit_TL[3]),float(v_limit_TL[2])).overlaps(pd.Interval(0,50/3.6)):
                fut_g = [fut_TL[2],fut_TL[3]]
                v_limit = [max(v_limit_TL[3],0),min(v_limit_TL[2],50/3.6)]
                if svSpd >=v_limit[0] and svSpd<=v_limit[1]:
                    fTL = (1.5-0.5)/(v_limit[1]-v_limit[0])*svSpd+0.5-v_limit[0]/(v_limit[1]-v_limit[0])
                    # fTL = (1-0.5)/(v_limit[1]-v_limit[0])*svSpd+0.5-v_limit[0]/(v_limit[1]-v_limit[0])
                else:
                    fTL = -1
            else:
                fut_g = [fut_TL[0],fut_TL[1]]
                fTL = 0
        else:
            d_tl = 470-data[0,7]-dis
            fut_g = [self.SimGreenStartData[max(np.argwhere((1<self.SimPosData) & (self.SimPosData<=470)))[0]],
                     self.SimGreenEndData[max(np.argwhere((1<self.SimPosData) & (self.SimPosData<=470)))[0]]]
            fTL = self.SimScoreTLData[max(np.argwhere((1<self.SimPosData) & (self.SimPosData<=470)))[0]]
            # print('self.SimScoreTLData',self.SimScoreTLData)
            # print('passed, score_TL', fTL)
        self.SimScoreTLData[self.timeStep-1] = fTL
        self.SimGreenStartData[self.timeStep-1] = fut_g[0]
        self.SimGreenEndData[self.timeStep-1] = fut_g[1]
        acc_IDM, v_IDM = IDM(space_pre, speed_pre, delta_spd, step_duration, IVT, s_s)
        if 440<=dis+data[0,7]<470 and data[self.timeStep-2,7]+self.currentState[0]>470:
            if curr_TL is not 'green':
                acc_IDM, v_IDM=IDM(470-(dis+data[0,7]), speed_pre, speed_pre, step_duration,IVT_TL,s_TL) 
                
        dis_IDM = dis + self.currentState[1]*step_duration + 0.5*acc_IDM*step_duration**2
        
        # if dis+data[0,7]<470:
        self.currentState=[space, svSpd, relSpd, d_tl, fut_g[0]-time, fut_g[1]-time]
        # else:
        #     self.currentState=[space, svSpd, relSpd, 0, 0, 0]
        self.s = np.hstack((self.s[6:],self.currentState))
        

        #fJerk = -(jerk**2)/((4+4)/0.04)**2  # the maximum range is change from -3 to 3 in 0.1 s, then the jerk = 60
        # fJerk=-2*(abs(jerk)/((4+4)/0.04))**(1/4)#+0.5#2* best
        # fJerk = -2*(np.count_nonzero(self.SimJerkData[:self.timeStep-1] > 1.5|self.SimJerkData[:self.timeStep-1] < -1.5))/(self.timeStep-1)
        experienced_jerk = self.SimJerkData[:self.timeStep-1]
        fJerk = -5*np.count_nonzero(abs(experienced_jerk) > 1.5)/(self.timeStep-1)
        
        # if jerk<=5 and jerk>=-5:
        #     fJerk = -1/5*abs(jerk)+1
        # elif jerk>10 or jerk<-10:
        #     fJerk =-((jerk-10)**2)/((4+4)/0.04-10)**2
        # else:
        #     fJerk = 0
        
        #fAcc = - (action**2)/4**2
        fAcc = -4*(abs(action)/4)**(1/2)-4*(abs(jerk)/((4+4)/0.04))**(1/4)
        self.lastAction = action

        if self.TTC >= 0 and self.TTC <= self.TTC_threshold:
            TTC_violate+=1
            fTTC = 5*((self.TTC/self.TTC_threshold)**2-1)
            fTTC_total += 5*((self.TTC/self.TTC_threshold)**2-1)
            #fTTC = np.log(self.TTC/self.TTC_threshold) 
        else:
            TTC_violate+=0
            fTTC_total += 0
            fTTC = 0
        # if TTC_violate !=0:
        #     fTTC= fTTC_total/TTC_violate
        # else:
            # fTTC = 0
        '''
        mu = 0.422618  
        sigma = 0.43659
        if hdw <= 0:
            fHdw = -1
        else:
            fHdw = (np.exp(-(np.log(hdw) - mu) ** 2 / (2 * sigma ** 2)) / (hdw * sigma * np.sqrt(2 * np.pi)))
        '''
        self.traveledTime = (data[self.timeStep-1,5]-data[0,5])
        fHdw =4*(dis / (data[self.timeStep-1,5]-data[0,5]))**2/(50/3.6)**2 #(dis / (data[self.timeStep-1,5]-data[0,5]))/(50/3.6)#
        try:
            fFuel = 6*(math.log(dis/(total_fuel*1000)+1,35))**5#25 30
        except:
            print('dis,total_fuel', dis,total_fuel)
        fIDM = -(abs(acc_IDM - action)/8)**0.5*1.5   
        # fIDM = (-(abs(acc_IDM - action)/8)**0.5)
        # -(abs(v_IDM - svSpd)/(50/3.6))**0.5-(abs(dis_IDM - dis)/(50/3.6*step_duration)**0.5)
        # if IDM_apply:
        #     fIDM = -1
        # else:
        #     fIDM =0
        # fIDM = -1.5*(abs(del_a)/(4-(-4)))**0.5
        
        # calculate the reward
        if space<0:
            reward = fJerk+fAcc + fFuel +fTL +fHdw+fCol +fIDM
            # reward = -1
        else:
            reward = fJerk+fAcc + fFuel +fTL +fHdw+fTTC + fIDM
        #reward = fFuel+ fHdw
        # record reward info
        
        rewardInfo = [self.TTC, hdw, jerk, fTTC, fHdw, fJerk, fAcc, total_fuel, fFuel,fTL,dis,fCol,TTC_violate,fTTC_total,fIDM]
        
        # judge the end
        if self.timeStep == self.TimeLen or self.isCollision == 1:
            done = True
            a_bound_jerk = None
            a_bound_v = None
        else:
            step_duration_next = (data[self.timeStep,5]-data[self.timeStep-1,5])
            a_bound_jerk = [action-step_duration_next*100,action+step_duration_next*100] 
            a_bound_v=[-self.currentState[1]/step_duration_next,(50/3.6-self.currentState[1])/step_duration_next]
            done = False
        s_=self.s

        return s_, reward, done, rewardInfo ,a_bound_jerk,a_bound_v