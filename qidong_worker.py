#!/usr/bin/env python3

"""
Created on Mon Dec 18 11:43:35 2023

@author: dongh
"""

from MU_EV.worker import single_worker
from MU_EV.controller.start_empgpe import MubotPGPE as pgpe
import os,rospkg,shutil, sys


os.environ['ROSCONSOLE_CONFIG_FILE'] = '/dev/null'  # disables config
os.environ['ROSCONSOLE_MIN_SEVERITY'] = 'WARN'
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('des_mubot')
    database = os.path.join(pkg_path,'Database')
    wk = single_worker(1,pkg_path)    
    if len(sys.argv)>1:
        if sys.argv[1]=='b':
            print("no illustration worker")
            wk.Evolution_start(base=True)
        elif sys.argv[1]=='c':
            print("Continue unfinished worker")
            wk.Evolution_continue()
        elif sys.argv[1]=='g':
            print("Generate gait for best individuals")
            wk.generate_gait()
        else:
            print("Unknown argument")
    else:
        print("Start new worker")
        wk.Evolution_start(base=None)

    # if len(sys.argv)>1:
    #     if sys.argv[1]=='c':
    #         print("Continue unfinished worker")
    #         wk.Evolution_continue()
    #     elif sys.argv[1]=='g':
    #         print("Generate gait for best individuals")
    #         wk.generate_gait()
    #     else:
    #         print("Unknown argument")
    # else:
    #     print("Start new worker")
    #     wk.Evolution_start(base=True)
    # wk.controller_optimize(act_num=4)
    # wk.restart_controller_optimization(morph_ID=1)
