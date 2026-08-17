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

class Controller_Template:
    def __init__(self,root,episod,time_step,init_path=None):
        # self.root = root
        self.joint_dict={}
        for joint in root.findall('joint'):
            self.joint_dict[joint.attrib['name']]=joint
        self.resis = cfg.ACT.RESISTANCE
        self.max_f_amp = cfg.ACT.MAX_F_AMP
        self.max_v_amp = cfg.ACT.MAX_V_AMP
        self.max_f_sig = cfg.ACT.MAX_F_SIG
        self.max_v_sig = cfg.ACT.MAX_V_SIG
        self.path=""
        self.act_num = len(self.joint_dict.keys())
        self.epi_num = episod
        self.time_step = time_step
        
    def forw_sine_generate(self):
        """For each actuator, V_i=a_i*sin(2*pi*f*t+p_i)
           sum(P_i) =P_total 
        """
        self.forw_control={}
        self.forw_control['fre'] = (float('%.4g'%random.uniform(0,self.max_f_amp)),0)
        for key in self.joint_dict.keys():
            self.forw_control[key] = (float('%.4g'%random.uniform(0,self.max_v_amp)),float('%.4g'%random.uniform(0,1)))            
        df = pd.DataFrame.from_dict(self.forw_control, orient='index')
        df.to_csv(self.path+"forward_signal.csv",header=False)
        return
    
    def back_sine_gene(self):
        """For each actuator, V_i=a_i*sin(2*pi*f*t+p_i)
           sum(P_i) =P_total 
        """
        self.back_control={}
        self.back_control['fre'] = (float('%.4g'%random.uniform(0,self.max_f_amp)),0)
        for key in self.joint_dict.keys():
            self.back_control[key] = (float('%.4g'%random.uniform(0,self.max_v_amp)),float('%.4g'%random.uniform(0,1)))            
        df = pd.DataFrame.from_dict(self.back_control, orient='index')
        df.to_csv(self.path+"backward_signal.csv",header=False)
        return
    
    def brak_gene(self):
        """For each actuator, 
           chose a time T(t0~t0+1/f) to stop,
           V_i= V0+kt or sign(k)*Vmax
           sum(P_i) =P_total 
        """
        self.brake_control={}
        self.brake_control['fre'] = (float('%.4g'%random.uniform(0,1)),0)
        for key in self.joint_dict.keys():
            self.brake_control[key] = (float('%.4g'%random.uniform(-100,100)),0)            
        df = pd.DataFrame.from_dict(self.brake_control, orient='index')
        df.to_csv(self.path+"brake_signal.csv",header=False)
        return
    
    def C_start_gene(self):
        """For each actuator, V_i=a_i*sin(2*pi*f*t+p_i)
           sum(P_i) =P_total 
        """
        self.cstart_control={}
        self.cstart_control['fre'] = (0,0,0,0,0)
        for key in self.joint_dict.keys():
            self.cstart_control[key] = (float('%.4g'%random.uniform(0,1)),float('%.4g'%random.uniform(-self.max_v_amp,self.max_v_amp)),float('%.4g'%random.uniform(0,1)),float('%.4g'%random.uniform(0,1)),float('%.4g'%random.uniform(-self.max_v_amp,self.max_v_amp)),)            
        df = pd.DataFrame.from_dict(self.back_control, orient='index')
        df.to_csv(self.path+"cstart_signal.csv",header=False)
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
        
        