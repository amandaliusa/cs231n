import sys
import git

repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
    
# setting path
sys.path.append(homedir)

import argparse
import os
import numpy as np
import pandas as pd
import pickle
import traceback
from utils import *

def get_metrics(output_path, show_plots):
    num_joints = 17

    # Get a list of all subjects
    subjects = os.listdir("../pretrained_models/poseformer/")

    skipped = []
    all_res = []

    # Convert frames to a numpy array
    for subjectid in subjects:
        print(subjectid)
        try:
            if (subjectid != "kunkun" and os.path.isdir("../pretrained_models/poseformer/{}".format(subjectid))):
                processed_npz_path="../pretrained_models/poseformer/{}/input_2D/".format(subjectid)
                res = np.load("{}/keypoints.npz".format(processed_npz_path))['reconstruction']
                res = np.reshape(res, (res.shape[1], res.shape[2], res.shape[3]))
                
                reshaped_res = np.ones((res.shape[0], num_joints * 3))
                mask_x = [i for i in range(0, num_joints*3, 3)]
                mask_y = [i+1 for i in range(0, num_joints*3, 3)]
                reshaped_res[:,mask_x] = res[:,:,0]
                reshaped_res[:,mask_y] = res[:,:,1]

                results = process_subject_poseformer(subjectid, reshaped_res, framerate=30, show_plots=show_plots)
                if results != None:
                    all_res.append(results) 
        except Exception as e:
            traceback.print_exc()
            skipped.append(subjectid)
            print("Skipped " + subjectid)
            continue
        
    print(skipped)
    print("skipped " + str(len(skipped)))

    # save results
    res_df = pd.DataFrame(all_res)
    res_df.to_csv("../results/results-poseformer-{}.csv".format(output_path)) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='todays_date', help='input output path')
    parser.add_argument('--show_plots', type=bool, default=False, help='input whether to generate')
    args = parser.parse_args()

    get_metrics(args.output_path, args.show_plots)


