import numpy as np
import pandas as pd
import os shutil



if __name__ == "__main__":
    rewrite_reward
    

def rewrite_reward():
    path = os.path.join(os.getcwd(),'Population_Data')
    old_reward = pd.read_csv(path+'/Population_Reward.csv')
    pop_list = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]
    for pop in pop_list:
        reward_path = os.path.join(path,pop,'reward.csv',index_col=False,header=None)
        if os.path.exists(reward_path):
            df = pd.read_csv(reward_path)
            if 'average_speed' in df.columns:
                df.rename(columns={'average_speed':'reward'},inplace=True)
                df.to_csv(reward_path,index=False)
                print(f"Rewrote reward for {pop}")