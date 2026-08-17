# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:34:18 2023

@author: dongh
"""
import os,random,sys,copy
import numpy as np
from lxml import etree
import pandas as pd
##
from MU_EV.utils import xml
from MU_EV.config import cfg
from MU_EV.utils import file as fu
from MU_EV.utils import Hydro_param as hyp

class Controller:
    def __init__(self,act_num,episoid,time_step):
        self.act_num = act_num
        self.epi_num = episoid
        self.time_step = time_step
        return
    
    def sampling(self,mean,sigma):
        samp_len = len(mean)
        sampled = np.zeros((samp_len,self.epi_num))
        for i in range(samp_len):
            sampled[i,:] = np.random.normal(mean[i],sigma[i],self.epi_num)
        return sampled
        
    def sine_wave(self,amp,pha,f,epi,stop_time):
        t = np.arange(0,stop_time,self.time_step)
        sine_vol = np.zeros((len(amp),len(t)))
        for i in range(len(amp)):
            sine_vol[i,:]=amp[i]*np.sin(2*np.pi*(f*t+pha[i]))
        return sine_vol
    
    def step_wave(self,time,amp):
        t = np.arange(0,time[0,-1],self.time_step)
        tp = np.array(time/self.time_step).astype(int)
        step_vol = np.zeros((len(amp[0,:]),len(t)))
        for i in range(len(amp[0,:])):
            step_vol[i,tp[0]:tp[1]]=amp[0]*np.ones((1,tp[1]-tp[0]-1))
            step_vol[i,tp[2]:tp[3]]=amp[1]*np.ones((1,tp[2]-tp[3]-1))
        return step_vol
    
    def PPO_update(self):
        return
        
        
        