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

def get_metrics(output_path, show_plots, get_3d):
    num_joints = 17

    # Get a list of all subjects
    subjects = os.listdir("../pretrained_models/poseformer/")

    skipped = []
    all_res = []

    to_skip = ['kunkun', 'woyjGsIt', 'ZAn8qhAb', 'mxKQbPdW', 'GITsdVy7', 'OuYG4U64', 'k3YTjMU4', 'yzutrWiF', 'l230IPIW', 
           'x77WJ10Q', 'XVvERXzh', 'oSFbRH4g', 'LGqwZPty', 'IMG_1623', 'JNh9mTC7', 'tFbOCnVv', '20230505_131419', 
           'ztKJoXiw', 'VID_20230505_201438080', 'gZry6Nal', 'trim.EB638E4E_B3DC_470E_883A_26253AC03B55', 'IMG_0067', 
           'VKW2KzGC', 'hsX5kAeZ', 'trim.BBF38598_A21A_4AFB_8BD6_EE7D6564DAED', 'F2FtnWVJ', 'nGyTDr5q', 'HAKCEJWo', 
           'Cw8bz9tw', 'uwtiYEpw', 'IMG_0822', 'j3iTbhie', 'IMG_8762', 'X5kmXm2t', 'JOmsdP6A', 'g1eKN8fK', 'UmopdCFg', 
           'zkO4XPXQ', 'IMG_5384', 'Wyg30nvk', 'iJsDlll8', 'IMG_6494', '2SV6hYB2', '0B2dfO4b', 'ku64Cwle', 'IMG_3013', 
           'video', 'yt2AP5Lo', 'iVGPuONf', 'KneeVideo_Seese_May8', 'eOBg4mwH', 'trim.495B207A_D816_405F_91CE_21FF0E67F907', 
           'IMG_1070', 'RJVU9j8P', '4RF13po6', 'MeABXs62', 'IMG_1202', 'BvPM57u7', '15IPa5iS', 'obDMup1i', 'g5sxqMgO', 
           'VID_20230507_175427', 'ESzlIzyO', 'IMG_0388', 'VEBURFuk', 'CWf3eyvo', 'QNEd2eGX', 'ZR3QAZmu', '0nUjlcd7', 
           'H1SDVD0X', 'IMG_4697', 'F6PuVthQ', 'qpxWRSOP', 'iqQ28Zc4', 'xWZi5URn', 'PA4bXBlr']

    # Convert frames to a numpy array
    for subjectid in subjects:
        print(subjectid)
        try:
            if (subjectid not in to_skip and os.path.isdir("../pretrained_models/poseformer/{}".format(subjectid))):
                processed_npz_path_2d="{}/pretrained_models/poseformer/{}/input_2D/".format(homedir, subjectid)
                res2d = np.load("{}/keypoints.npz".format(processed_npz_path_2d))['reconstruction']
                res2d = np.reshape(res2d, (res2d.shape[1], res2d.shape[2], res2d.shape[3]))
                
                reshaped_res_2d = np.ones((res2d.shape[0], num_joints * 3))
                mask_x = [i for i in range(0, num_joints*3, 3)]
                mask_y = [i+1 for i in range(0, num_joints*3, 3)]
                reshaped_res_2d[:,mask_x] = res2d[:,:,0]
                reshaped_res_2d[:,mask_y] = res2d[:,:,1]

                res3d = None
                if get_3d:
                    processed_npy_path_3d="{}/pretrained_models/poseformer/{}/".format(homedir, subjectid)
                    res3d = np.load("{}/3d_output.npy".format(processed_npy_path_3d))

                results = process_subject_poseformer(subjectid, reshaped_res_2d, res3d, framerate=30, show_plots=show_plots)

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
    parser.add_argument('--get_3d', type=bool, default=True, help='input whether to generate 3d metrics')
    args = parser.parse_args()

    get_metrics(args.output_path, args.show_plots, args.get_3d)