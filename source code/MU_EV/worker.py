
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:43:35 2023

@author: donghao li
"""

from MU_EV.Morphology_Generate import MUBot_morph as MG
# from MU_EV.utils import controller_generation as cg
from MU_EV.config import cfg
from MU_EV.controller.start_empgpe import MubotPGPE as stpgpe
from MU_EV.controller.start_empgpe_base import MubotPGPE as stpgpe_base
from MU_EV.utils import file as fu
import numpy as np
import pandas as pd
import rosgraph, rospy
import sys, os, shutil, time, pathlib
from datetime import datetime

class single_worker:
    def __init__(self,worker_id,project_path):
        rospy.logwarn("worker initializing...")
        self.wk_id = worker_id
        self.project_path = project_path  # /home/donghaoli/test/src/des_mubot
        self.ROS_MASTER_URI    = 10000
        self.GAZEBO_MASTER_URI = 20000        
        self.population_path = project_path+'/Population_data'
        self.tornament_size  = cfg.EVO.TOR_PARTICIPANTS
        self.population_goal = cfg.EVO.POPULATION_GOAL
        self.population_init = cfg.EVO.INIT_POPULATION_SIZE
        self.aging_size      = cfg.EVO.AGING_WINDOW_SIZE 
        self.cwd = os.getcwd()
        self.worker_folder = self.project_path+'/Database/'+str(self.wk_id)
        self._check_worker_ID_duplicate()
        ####################################
        log_dir = f"/dev/shm/roslog_worker_{self.wk_id}"
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        os.environ["ROS_LOG_DIR"] = log_dir
        os.environ["ROSCONSOLE_CONFIG_FILE"] = "/dev/null"
        ##########################################
        os.environ["ROS_MASTER_URI"]="http://localhost:"+str(self.wk_id+self.ROS_MASTER_URI)
        os.environ["GAZEBO_MASTER_URI"]="http://localhost:"+str(self.wk_id+self.GAZEBO_MASTER_URI)
        os.environ["HEADLESS"] = "1"
        os.environ.pop("DISPLAY", None)
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["GAZEBO_HEADLESS_RENDERING"] = "1"
        os.environ["ROS_LOG_LEVEL"]            = "WARN"
        os.environ["ROS_MASTER_URI_TIMEOUT"]  = "5"
        os.environ["ROSCPP_CONN_TIMEOUT"]     = "5"
        ##########################################################################
        
        
        return

    def _check_worker_ID_duplicate(self):
        while rosgraph.is_master_online(master_uri='http://localhost:'+str(10000+self.wk_id)) or  os.path.exists(self.worker_folder):
            rospy.logwarn('Worker ID_'+str(self.wk_id)+' already in use, try next ID')
            self.wk_id += 1
            self.worker_folder = self.project_path+'/Database/'+str(self.wk_id)            
            if self.wk_id>200:
                raise Exception("Too many Gazebo and ROS Running!!")
        if not os.path.exists(self.worker_folder):
            os.mkdir(self.worker_folder)
        rospy.logwarn('Worker ID_'+str(self.wk_id)+' is assigned')
        return
    
    def _check_population_size(self):
        if not os.path.exists(self.population_path):
            os.mkdir(self.population_path)
            population_size = 0
        else:
            population_list = [f for f in os.listdir(self.population_path) if os.path.isdir(os.path.join(self.population_path,f))]
            population_size = len(population_list)
        return population_size
    
    def Evolution_start(self, base=True):
        while self._check_population_size() < self.population_init:
            rospy.logerr('\n\nEvolution Init, size: '+str(self._check_population_size()))
            self._evolution_init(base=base)
        while self._check_population_size() < self.population_goal:
            rospy.logerr('\n\nEvolution start, size: '+str(self._check_population_size()))
            self._evolution_work(base=base)
        rospy.logerr('Population goal reached')
        return True

    def _controller_optimization(self, morph_ID, base=True):
        print("controller optimization")
        max_sim_time = cfg.EVO.SIM_TIME
        with open(self.worker_folder+'/morph_ID.txt','w') as f_id:
            f_id.write(str(morph_ID))
        if not base:
            pgpe = stpgpe(self.worker_folder, max_sim_time, self.wk_id, morph_ID)
        else:
            pgpe = stpgpe_base(self.worker_folder, max_sim_time, self.wk_id, morph_ID)

        df_temp = pd.read_csv(self.worker_folder+'/joint_param.csv',header=None,index_col=0)
        act_num = df_temp.shape[0]
        reward,cpg = pgpe.run_pgpe(self.wk_id, act_num)

        ### save results
        try:
            shutil.move(self.worker_folder+'/cpg_history.csv', self.population_path+'/'+str(morph_ID)+'/cpg_history.csv')
            shutil.move(self.worker_folder+'/rollout_history.csv', self.population_path+'/'+str(morph_ID)+'/rollout_history.csv')
            shutil.move(self.worker_folder+'/reward_history.csv', self.population_path+'/'+str(morph_ID)+'/reward_history.csv')
            shutil.move(self.worker_folder+'/drift_history.csv', self.population_path+'/'+str(morph_ID)+'/drift_history.csv')
        except FileNotFoundError:
            pass
        return reward,cpg


    def _evolution_work(self,base=True):
        print('Evolving')
        parent_ID = self.tornament()  # got parent ID
        # 
        morph_parent = MG(self.project_path,self.wk_id,parent_ID)
        child_ID = morph_parent.mutation()  ## save in Population_data/morph_ID and Database/worker_ID
        print(f"\33[92mGenerated child ID: {child_ID}\033[0m")
        reward,cpg = self._controller_optimization(child_ID,base=base)
        self.update_population(child_ID,reward,cpg)
        return
    
    def _evolution_init(self, base = None):
        morph_parent = MG(self.project_path,self.wk_id,None,new_morph=True)        
        morph_ID = morph_parent.morph_ID
        # print('Generating Initial MUBot ID: '+str(morph_parent.morph_ID))
        reward,cpg = self._controller_optimization(morph_ID,base=base)
        # print('Evolution Init ID: '+str(new_ID))
        self.update_population(morph_ID,reward,cpg)
        return

    def tornament(self):
        ## open population database and 
        population_file = self.population_path+'/Population_Reward.csv'
        temp = pd.read_csv(population_file,header=None,index_col=None,sep=',').iloc[:,:2]
        all_df = temp.tail(self.aging_size)
        tor_df = all_df.sample(n=self.tornament_size).sort_values(by=[1])
        winner_ID = tor_df.iloc[-1][0]
        return winner_ID
    
    def update_population(self,morph_ID,reward,cpg):
        population_file = self.population_path+'/Population_Reward.csv'
        os.makedirs(self.project_path + '/Population_data/' + str(morph_ID), exist_ok=True)
        pd.DataFrame(cpg).T.to_csv(self.project_path+'/Population_data/'+str(morph_ID)+'/controller.csv',header=None,index=None)
        mode = 'w' if not os.path.exists(population_file) else 'a'
        finish_time = round(datetime.now().timestamp(),6)
        print(reward)
        pop_info = np.array([str(morph_ID) + ',' + f"{reward:.8e}"  + ',' + str(finish_time)], dtype=str)
        with open(population_file, mode) as f_up:
            np.savetxt(f_up, pop_info, fmt='%s')
        return

    def Evolution_continue(self):
        #moving unfinished worker data to worker folder
        uf_list = os.listdir(self.project_path+'/Unfinished_worker')
        while len(os.listdir(self.project_path+'/Unfinished_worker')) > 0:
            uf_list = os.listdir(self.project_path+'/Unfinished_worker')
            # rospy.logerr("morph remaining to continue: "+str(len(uf_list)))
            shutil.move(self.project_path+'/Unfinished_worker/'+str(uf_list[0]), self.worker_folder)
            fu.move_files(self.worker_folder+'/'+uf_list[0],self.worker_folder)        
            os.rmdir(self.worker_folder+'/'+uf_list[0])
            # continue controller optimization
            morph_ID = uf_list[0]
            max_sim_time = cfg.EVO.SIM_TIME
            print(f"Continue Evolution for morph ID: {morph_ID}")
            pgpe = stpgpe_base(self.worker_folder, max_sim_time, self.wk_id,morph_ID)
            df_temp = pd.read_csv(self.worker_folder+'/joint_param.csv',header=None,index_col=0)
            act_num = df_temp.shape[0]
            # reward,cpg = pgpe.continue_pgpe(self.wk_id, act_num)
            reward,cpg = pgpe.continue_pgpe(self.wk_id, act_num)
            self.update_population(morph_ID,reward,cpg)

            
            ### save results
            os.makedirs(self.population_path+'/'+str(morph_ID), exist_ok=True)
            fu.move_files(self.worker_folder, self.population_path+'/'+str(morph_ID))
            rospy.logerr('Finished Evolution')
        return True
    
    def generate_gait(self):
        reward_list = pd.read_csv(self.population_path+'/Population_Reward.csv',header=None,index_col=None)
        reward_sort = reward_list.sort_values(by=[1],ascending=False)#.iloc[100:]#[2:3]
        print(reward_sort)
        for index, row in reward_sort.iterrows():
            morph_ID = row[0]
            # print('Generating gait for morph ID: '+str(morph_ID))
            morph_folder = self.population_path+'/'+str(morph_ID)
            gait_folder = self.project_path+'/Gait/'+str(morph_ID)
            # os.makedirs(gait_folder, exist_ok=True)
            print("Count: "+str(index)+"  "+str(self.worker_folder))
            shutil.copytree(morph_folder, self.worker_folder, dirs_exist_ok=True)
            max_sim_time = cfg.EVO.SIM_TIME
            pgpe = stpgpe_base(self.worker_folder, max_sim_time, self.wk_id)
            df_temp = pd.read_csv(self.worker_folder+'/joint_param.csv',header=None,index_col=0)
            act_num = df_temp.shape[0]
            reward = pgpe.one_pepg(self.wk_id, morph_ID,act_num)
            # pd.DataFrame([morph_ID,reward]).to_csv(gait_folder+'/gait_reward.csv',header=None,index=None)
            # pd.DataFrame([morph_ID,reward]).to_csv(self.project_path+'/Gait/r_'+str(morph_ID)+'.csv',header=None,index=None)
            # shutil.move(self.worker_folder+'/gait_record.csv', self.project_path+'/Gait/g_'+str(morph_ID)+'.csv')
            shutil.move(self.worker_folder+'/displace_record.csv', self.project_path+'/Gait/d_'+str(morph_ID)+'.csv')
            # shutil.move(self.worker_folder+'/hydro_record.csv', gait_folder+'/hydro_record.csv')
            # shutil.copytree(self.worker_folder, gait_folder, dirs_exist_ok=True)
        return
            # shut


    
