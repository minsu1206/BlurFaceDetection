import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from loss import *
##########################################################
#                  Functions about Loss                  #
##########################################################

def build_loss_func(loss_dict, device):

    loss_compute_dict = {}

    for key, val in loss_dict.items():
        key = key.lower()
        
        func = None
        # Task : Regression
        if key == 'huber':
            func = HuberLoss(device=device)

        if key == 'mse':
            func = nn.MSELoss()

        if key == 'l1':
            func = nn.L1Loss()

        if key == 'probbasedmse':
            func = ProbBasedMSE(device=device)

        # Task : Classification
        if key == 'crossentropy':
            func = nn.CrossEntropyLoss()

        if func == None:
            raise NotImplementedError(f"{key} is not implemented yet.")
        weight = val
    
        loss_compute_dict[key] = {'func': func.to(device), 'weight': weight}

    return loss_compute_dict


def compute_loss(loss_func, pred, gt_reg, gt_cls):
    total_loss = 0

    assert pred.get_device() == gt_reg.get_device(), \
        "Prediction & GT tensor must be in same device"
    if gt_cls != None:
        assert pred.get_device() == gt_cls.get_device(), \
            "Prediction & GT tensor must be in same device"

    for loss_name, loss_dict in loss_func.items():

        if loss_name in ['crossentropy']:
            loss = loss_dict['func'](pred, gt_cls)
        else:
            loss = loss_dict['func'](pred, gt_reg)

        loss *= loss_dict['weight']
        total_loss += loss
    
    return total_loss

##########################################################
#                  Functions about Optim                 #
##########################################################
def build_optim(cfg, model):
    optim_name = cfg['train']['optim']
    lr = cfg['train']['lr']
    optim_name = optim_name.lower()

    optim = None
    if optim_name == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    if optim_name == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    if optim_name == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Add optimizer if you want

    if optim != None:
        return optim
    else:
        raise NotImplementedError(f"{optim_name} is not implemented yet.")


##########################################################
#                  Functions about Scheduler             #
##########################################################
def build_scheduler(cfg, optimizer):
    scheduler_dict = cfg['train']['scheduler']

    sch_name = list(scheduler_dict.keys())[0]
    sch_settings = scheduler_dict[sch_name]
    sch_name = sch_name.lower()

    if sch_name == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sch_settings['milestones'], gamma=sch_settings['gamma']
        )
    
    if sch_name == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=sch_settings['gamma']
        )

    if sch_name == 'cossineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sch_settings['T_max'], eta_min=sch_settings['eta_min']
        )

    # Add optimizer if you want

    if sch_name != None:
        return scheduler
    else:
        raise NotImplementedError(f"{optim_name} is not implemented yet.")


##########################################################
#                       Visualize                        #
##########################################################

def visualize(model, input_size, device, epoch, save_path):
    
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

    model.eval()
    with torch.no_grad():
        
        cos_mean_random = np.zeros(100)
        cos_mean_fix = np.zeros(100)

        with open(random_pkl_path, 'rb') as f:
            real_mean_random = pickle.load(f)

        with open(fix_pkl_path, 'rb') as f:
            real_mean_fix = pickle.load(f)
        
        print(f"Epoch #{epoch + 1 } >>>> Visualize :")

        iter = 0
        for subpath in tqdm(os.listdir(path)):
            if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
                sample_path = os.path.join(path, subpath)
                iter += 1
                for i in range(1, 101):
                    try:  
                        blurred_img_fix = cv2.resize(cv2.imread(os.path.join(sample_path, f'fix_{i}.png')),
                                                        (input_size, input_size), interpolation=cv2.INTER_AREA) / 255
                        blurred_img_random = cv2.resize(cv2.imread(os.path.join(sample_path, f'random_{i}.png')),
                                                        (input_size, input_size), interpolation=cv2.INTER_AREA) / 255

                        blurred_img_fix = torch.Tensor(blurred_img_fix).permute(2, 0, 1).unsqueeze(0).to(device)
                        estimated_fix = model(blurred_img_fix)
                        blurred_img_random = torch.Tensor(blurred_img_random).permute(2, 0, 1).unsqueeze(0).to(device)
                        estimated_random = model(blurred_img_random)

                    except:
                        estimated_random = (cos_mean_random[i - 1] / iter)
                        estimated_fix = (cos_mean_fix[i - 1] / iter)
                        
                    cos_mean_random[i - 1] += estimated_random.item()
                    cos_mean_fix[i - 1] += estimated_fix.item()
                
            break

        cos_mean_fix /= 30
        cos_mean_random /= 30

        plt.figure(figsize=(24, 7))
        plt.subplot(1, 2, 1)
        plt.plot(cos_mean_fix, 'k', linewidth=2, label="Estimated(Fix $\\theta$)")
        plt.plot(real_mean_fix, 'k--', linewidth=2, label="Real(Fix $\\theta$)")
        plt.legend(fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(cos_mean_random, 'k', linewidth=2, label="Estimated(Random $\\theta$)")
        plt.plot(real_mean_random, 'k--', linewidth=2, label="Real(Random $\\theta$)")
        plt.legend(fontsize=15)
        plt.savefig(f"{save_path}/graph_{epoch}.png")

    return cos_mean_fix, cos_mean_random

def visualize_cls(model, input_size, device, epoch, cls_num, save_path):
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

    model.eval()
    with torch.no_grad():
        
        cos_mean_random = np.zeros(100)
        cos_mean_fix = np.zeros(100)

        with open(random_pkl_path, 'rb') as f:
            real_mean_random = pickle.load(f)

        with open(fix_pkl_path, 'rb') as f:
            real_mean_fix = pickle.load(f)

        iter = 0
        for subpath in tqdm(os.listdir(path)):
            if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
                sample_path = os.path.join(path, subpath)
                iter += 1
                for i in range(1, 101):
                    try:
                        blurred_img_fix = cv2.resize(cv2.imread(os.path.join(sample_path, f'fix_{i}.png')),
                                                     (input_size, input_size), interpolation=cv2.INTER_AREA) / 255
                        blurred_img_random = cv2.resize(cv2.imread(os.path.join(sample_path, f'random_{i}.png')),
                                                        (input_size, input_size), interpolation=cv2.INTER_AREA) / 255

                        blurred_img_fix = torch.Tensor(blurred_img_fix).permute(2, 0, 1).unsqueeze(0).to(device)
                        estimated_fix = model(blurred_img_fix)
                        _, predicted_fix = torch.max(estimated_fix.data, 1)
                        
                        blurred_img_random = torch.Tensor(blurred_img_random).permute(2, 0, 1).unsqueeze(0).to(device)
                        estimated_random = model(blurred_img_random)
                        _, predicted_random = torch.max(estimated_random.data, 1)
                        
                        estimated_random = predicted_random*(1/cls_num)
                        estimated_fix = predicted_fix*(1/cls_num)

                    except:
                        estimated_random = (cos_mean_random[i - 1] / iter)
                        estimated_fix = (cos_mean_fix[i - 1] / iter)

                    cos_mean_random[i - 1] += estimated_random.item()
                    cos_mean_fix[i - 1] += estimated_fix.item()

        cos_mean_fix /= 30
        cos_mean_random /= 30

        plt.figure(figsize=(24, 7))
        plt.subplot(1, 2, 1)
        plt.plot(cos_mean_fix, 'k', linewidth=2, label="Estimated(Fix $\\theta$)")
        plt.plot(real_mean_fix, 'k--', linewidth=2, label="Real(Fix $\\theta$)")
        plt.legend(fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(cos_mean_random, 'k', linewidth=2, label="Estimated(Random $\\theta$)")
        plt.plot(real_mean_random, 'k--', linewidth=2, label="Real(Random $\\theta$)")
        plt.legend(fontsize=15)
        plt.savefig(f"{save_path}/graph_{epoch}_cls_{cls_num}_visualize.png")
        plt.close()

    return cos_mean_fix, cos_mean_random