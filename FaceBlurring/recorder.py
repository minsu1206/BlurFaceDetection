import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import pytorch_model_summary
import torchvision.transforms as transforms

import glob
from dataset.dataset import FaceDataset, FaceDatasetVal
from utils import *
from model_factory import model_build
from loss import *


COLOR_LIST = [
    'r','seagreen', 'deepskyblue', 'orange', 'gold', 
    'greenyellow', 'royalblue', 'indigo', 'magenta',
    'silver', 'gray', 'k'
]


def recorder(args):

    '''Function for recording some models' results'''
    ##########################################################
    #                     configuration                      #
    ##########################################################

    # (0) : Test set 
    path = 'data_samples/samples'
    if not os.path.exists(path):
        path = os.getcwd() + path
    
    random_pkl_path = 'data_samples/random_reference.pkl'
    if not os.path.exists(random_pkl_path):
        random_pkl_path = os.getcwd() + random_pkl_path

    fix_pkl_path = 'data_samples/fix_reference.pkl'
    if not os.path.exists(fix_pkl_path):
        fix_pkl_path = os.getcwd() + fix_pkl_path

    assert os.path.exists(path) == True
    assert os.path.exists(random_pkl_path) == True
    assert os.path.exists(fix_pkl_path) == True

    with open(random_pkl_path, 'rb') as f:
        real_mean_random = pickle.load(f)

    with open(fix_pkl_path, 'rb') as f:
        real_mean_fix = pickle.load(f)

    suffix = '$\\theta$'
    fix_plots = {'Real Fix ' + suffix: real_mean_fix}
    random_plots = {'Real Random '+ suffix: real_mean_random}


    # (1) : Bring Model results 
    for checkpoint in args.checkpoints:

        one_fix_pkl = list(glob.glob(checkpoint + '/*_cos_mean_fix.pkl'))[0]
        one_random_pkl = list(glob.glob(checkpoint + '/*_cos_mean_fix.pkl'))[0]
        exp_name = os.path.basename(one_fix_pkl).replace('_cos_mean_fix.pkl', '')
        
        with open(one_fix_pkl, 'rb') as f:
            cos_mean_fix = pickle.load(f)
        
        with open(one_random_pkl, 'rb') as f:
            cos_mean_random = pickle.load(f)
        
        fix_plots[exp_name + ' ' +suffix] = cos_mean_fix
        random_plots[exp_name + ' ' +suffix] = cos_mean_random
    
    # (2) : Plot Model Results and Show Metric as Table
    plt.figure(figsize=(24, 7))
    plt.subplot(1, 2, 1)
    for i, (key, val) in enumerate(fix_plots.items()):
        plt.plot(val, linewidth=2, label=key, color=COLOR_LIST[i])
    plt.legend(fontsize=15)

    plt.subplot(1, 2, 2)
    for i, (key, val) in enumerate(random_plots.items()):
        plt.plot(val, linewidth=2, label=key, color=COLOR_LIST[i])
    plt.legend(fontsize=15)

    plt.savefig(args.save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs="+")
    parser.add_argument('--save_path', type=str, help='.png or .jpg at last')
    args = parser.parse_args()

    """
    args.checkpoints = [dir1, dir2, dir2, ...]
    
    folder structure should be like ...

    dir1
        - {exp_name_1}_cos_mean_fix.pkl
        - {exp_name_1}_cos_mean_random.pkl
    dir2
        - {exp_name_2}_cos_mean_fix.pkl
        - {exp_name_2}_cos_mean_random.pkl
    ...

    These *_cos_mean_fix.pkl can be optained w/ test.py, after training model w/ train.py
    """
    recorder(args)
