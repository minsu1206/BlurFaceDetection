import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn

def build_loss_func(cfg):
    '''Return loss function'''
    model_name = cfg['train']['model']
    if model_name == 'resnet_cls':
        loss_name_list = [cfg['train']['loss1'], cfg['train']['loss2']]
        loss_func = []
        for loss_name in loss_name_list:
            if loss_name == 'huber':
                loss_func.append(nn.HuberLoss())

            if loss_name == 'cross_entropy':
                loss_func.append(nn.CrossEntropyLoss())

    else:
        loss_name = cfg['train']['loss']
        if loss_name == 'huber':
            loss_func = nn.HuberLoss()
    
    return loss_func

def build_optim(cfg, model):
    '''Return optimizer'''
    optim = cfg['train']['optim']
    optim = optim.lower()
    lr = cfg['train']['lr']

    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)

    if optim == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    
    # TODO : add optimizer

def build_scheduler(cfg, optim):
    '''Return learning rate scheduler'''
    scheduler_dict = cfg['train']['scheduler']
    scheduler, spec = scheduler_dict.items()
    scheduler = scheduler.lower()
    
    if scheduler == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=spec["milestones"], gamma=spec["gamma"])

    if scheduler == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(optim, base_lr=spec["base_lr"], max_lr=spec["max_lr"])
    
    # TODO : add leraning rate scheduler

def visualize(model, input_size, device, epoch):
    model.eval()
    with torch.no_grad():
        path = '/data/faceblur/BlurFaceDetection/FaceBlurring/data_samples/samples'
        cos_mean_random = np.zeros(100)
        cos_mean_fix = np.zeros(100)

        with open('/data/faceblur/BlurFaceDetection/FaceBlurring/data_samples/random_reference.pkl', 'rb') as f:
            real_mean_random = pickle.load(f)

        with open('/data/faceblur/BlurFaceDetection/FaceBlurring/data_samples/fix_reference.pkl', 'rb') as f:
            real_mean_fix = pickle.load(f)

        iter = 0
        for subpath in os.listdir(path):
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
        plt.savefig(f"graph_{epoch}.png")
