#!/usr/bin/env python

from MU_EV.controller.mubot_rob_env import MubotEnv # type: ignore
from MU_EV.controller.EM_pgpe import EMPGPE # type: ignore
import numpy as np
import rospy, rospkg
import pandas as pd
import random, os, time, sys, shutil

class MubotPGPE:
    def __init__(self,file_path, max_sim_time, wk_id):
        self.env = MubotEnv(file_path, max_sim_time, wk_id)
        self.file_path = file_path+"/"
        pass

    def one_pepg(self,wk_id,morph_ID,act_num):
        ###############################################################################################
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('des_mubot')
        self.wk_id = wk_id
    # ######################################################################################################          # number of actuators
        ### initialize PGPE parameters
        
        n_param   = act_num*2                               # number of parameter (amp, (amp+phase)*n, fre)
        temp_input = np.zeros(n_param+1)
        reward = 0.0
        #######################################################################################################
        # Start PGPE learning
        rospy.logwarn("##### Rerun for morph ID: "+str(morph_ID)+" #####")
        df = pd.read_csv(self.file_path+'/controller.csv',header=None,index_col=None).to_numpy().flatten()
        temp_input = df[:-1]
        ##### run Gazebo and get reward
        reward = self.env.step()
        rospy.logerr(f" Reward:  {reward*1000:.3e}")
        ###############################################################################################
        return
    
    def _csv_write(self,filename, data, mode='a'):
        df_data = pd.DataFrame(data.reshape(1,-1))
        # print(df_data.shape)
        df_data.to_csv(self.file_path+filename,float_format='%.8f',header=None,index=None,mode=mode)
        return
    
    def single_rollout(self,control_signal):
        self._csv_write('control_signal.csv', control_signal,mode='w')
        reward = self.env.step()
        rospy.logerr(f" Reward:  {reward*1000:.3e}")
        return reward

    def single_episode(self):
        return
