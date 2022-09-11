from typing import Iterable
import torch
import pandas as pd
import numpy as np
import math
import cv2, os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
#from blur import *

class FaceDatasetVal(Dataset):
    def __init__(
        self, csv_file, metric='cosine', transform=None, input_size=None, cmap='gray', 
        task='regression', num_classes=20):
        # [9/12 2AM]
        if hasattr(csv_file, '__iter__'):
            df_list = []
            for c in csv_file:
                df_list += [pd.read_csv(c)]
            self.path_n_label = pd.DataFrame.to_dict(pd.concat(df_list, ignore_index=True))
        else:
            self.path_n_label = pd.DataFrame.to_dict(pd.read_csv(csv_file))
            
        self.num_classes = num_classes
        self.paths, self.labels = self._get_training_samples()
        self.metric = metric
        assert metric in self.path_n_label.keys(), 'Not available metric, you have to create label'

        self.task = task

        if input_size is None:
            self.input_size = 1024
        else:
            self.input_size = input_size

        self.cmap = cmap
        
    def _get_training_samples(self):
        paths, labels = [], []
        for i in range(len(self.path_n_label['filename'])):
            if self.path_n_label['Unnamed: 0'][i]:
                paths.append('./BlurFaceDetection/FaceBlurring' + self.path_n_label['filename'][i][2:])
                labels.append(self.path_n_label['cosine'][i])

        return paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]

        try:
            if self.cmap == 'gray':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            elif self.cmap == 'rgb':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        except:
            print(img_path, ': Regenerate blurred sample')
            img_path, label = self.paths[idx+1], self.labels[idx+1]
            if self.cmap == 'gray':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            elif self.cmap == 'rgb':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        image = cv2.resize(image,
                           (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA)

        if self.transform:
            image = self.transform(image).float()

        if self.task == 'regression':
            label = torch.from_numpy(np.asarray(label)).float()
        else:
            label = (round(label * self.num_classes), torch.from_numpy(np.asarray(label)).float())

        return image, label

'''
def apply_more_blur(clean, blur, resnet, device):
    degree = 0.0
    clean, blur = cv2.imread(clean), cv2.imread(blur)
    patience = 10
    iteration = 0
    while degree < 0.63 and iteration <= patience:
        blur, _ = blurring(blur, {'mean':50, 'var':20, 'dmin':70, 'dmax':100})
        emb_clean = resnet(torch.Tensor(clean).permute(2, 0, 1).unsqueeze(0).to(device))
        emb_blur = resnet(torch.Tensor(blur).permute(2, 0, 1).unsqueeze(0).to(device))
        cosine = F.cosine_similarity(emb_clean, emb_blur, 1).item()
        degree = 1-cosine
        iteration += 1

    return blur, degree
'''

class FaceDataset(Dataset):
    def __init__(self, csv_file, metric='cosine', transform=None, input_size=None, cmap='gray', 
    task='regression', num_classes=20):
        if hasattr(csv_file, '__iter__'):
            df_list = []
            for c in csv_file:
                df_list += [pd.read_csv(c)]
            self.path_n_label = pd.DataFrame.to_dict(pd.concat(df_list, ignore_index=True))
        else:
            self.path_n_label = pd.DataFrame.to_dict(pd.read_csv(csv_file))

        self.num_classes = num_classes
        self.paths, self.labels = self._get_training_samples()
        self.metric = metric
        assert metric in self.path_n_label.keys(), 'Not available metric, you have to create label'
        self.transform = transform
        self.task = task

        if input_size is None:
            self.input_size = 1024
        else:
            self.input_size = input_size

        self.cmap = cmap

    def _get_training_samples(self):
        paths, labels = [], []
        for i in range(len(self.path_n_label['filename'])):
            if self.path_n_label['train'][i]:
                paths.append(self.path_n_label['filename'][i])
                labels.append(self.path_n_label['cosine'][i])

        return paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        try:
            if self.cmap == 'gray':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            elif self.cmap == 'rgb':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        except:
            print(img_path, ': Regenerate blurred sample')
            img_path, label = self.paths[idx+1], self.labels[idx+1]
            if self.cmap == 'gray':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
            elif self.cmap == 'rgb':
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        image = cv2.resize(image,
                           (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA)
        if self.transform:
            image = self.transform(image).float()

        if self.task == 'regression':
            label = torch.from_numpy(np.asarray(label)).float()

        else:
            # label = (round(label * self.num_classes), torch.from_numpy(np.asarray(label)).float())
            label = (torch.from_numpy(np.asarray(label)).float(), round(label * self.num_classes))

        return image, label

if __name__ == '__main__':
    ''' Code for generating blur samples again
    clean_pth = '../data/FFHQ_1024/clean/11000/11021.png'
    clean = cv2.imread(clean_pth)
    blurred, degree = blurring(clean, {'mean':50, 'var':20, 'dmin':0, 'dmax':100})
    cv2.imwrite('../data/FFHQ_1024/blur_defocus/11000/11021.png', blurred)

    resnet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
    emb_clean = resnet(torch.Tensor(clean).permute(2, 0, 1).unsqueeze(0).cuda())
    emb_blur = resnet(torch.Tensor(blurred).permute(2, 0, 1).unsqueeze(0).cuda())
    cosine = F.cosine_similarity(emb_clean, emb_blur, 1).item()

    dfdict = pd.DataFrame.to_dict(pd.read_csv('../data/label_blur_defocus/label/label.csv'))
    for i in range(len(dfdict['filename'])):
        if dfdict['filename'][i] == '../data/FFHQ_1024/blur_defocus/11000/11021.png':
            print(i, dfdict['filename'][i], dfdict['cosine'][i])
            print("Changing..")
            dfdict['cosine'][i] = 1-cosine
            print(i, dfdict['filename'][i], dfdict['cosine'][i])
    df = pd.DataFrame(dfdict)
    df.to_csv('../data/label_blur_defocus/label/label.csv')
    '''
    
    ###############
    dfdict = pd.DataFrame.to_dict(pd.read_csv('../data/label_defocus/label/label.csv'))
    labels1 = []
    for i in range(len(dfdict['filename'])):
        labels1.append(dfdict['cosine'][i])

    new_dict = {'filename' : [], 'cosine' : [], 'train' : []}
    cnt_samples = np.zeros(81)
    clean_paths, mblur_paths = [], []
    for i in tqdm(range(len(dfdict['filename']))):
        idx = math.trunc(dfdict['cosine'][i]/0.0125)
        if idx == 0:
            f = dfdict['filename'][i]
            mblur_paths.append(f)
            c = os.path.join((os.path.sep).join(f.split(os.path.sep)[:3]), 'clean', (os.path.sep).join(f.split(os.path.sep)[4:]))
            clean_paths.append(c)

        cnt_samples[idx] += 1

        if cnt_samples[idx] <= 700:
            new_dict['train'].append(True)
            new_dict['filename'].append(dfdict['filename'][i])
            new_dict['cosine'].append(dfdict['cosine'][i])
        else:
            new_dict['train'].append(False)
            new_dict['filename'].append(dfdict['filename'][i])
            new_dict['cosine'].append(dfdict['cosine'][i])

    plt.figure(figsize=(12, 7))
    plt.hist(labels1, bins=80, color='black')
    plt.show()
    ###############


    ''' Code for sub sample more blurry images
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reblur_clean, reblur_pair = clean_paths[2000:], mblur_paths[2000:]
    print('Reblur samples :',len(reblur_pair))
    blur_samples = []
    for c, b in tqdm(zip(reblur_clean, reblur_pair)):
        blur, degree = apply_more_blur(c, b, resnet=InceptionResnetV1(pretrained='vggface2', device=device).eval(), device=device)
        idx = new_dict['filename'].index(b)
        new_dict['cosine'][idx] = degree
        blur_samples.append(blur)

    print(reblur_pair[10])
    '''


    #########################################
    labels2 = []
    for i in range(len(new_dict['filename'])):
        if new_dict['train'][i]:
            labels2.append(new_dict['cosine'][i])

    plt.figure(figsize=(12, 7))
    plt.hist(labels2, bins=80, color='black')
    plt.show()

    answer = input("Press y if want to save or n")
    if answer == 'y':
        df = pd.DataFrame(new_dict)
        df.to_csv('../data/label_defocus/label/data_label.csv')
    '''
    dfdict1 = pd.DataFrame.to_dict(pd.read_csv('../data/label_random/label/data_label.csv'))
    dfdict2 = pd.DataFrame.to_dict(pd.read_csv('../data/label_defocus/label/data_label.csv'))
    labels3 = []
    for i in range(len(dfdict1['filename'])):
        if dfdict1['train'][i]:
            labels3.append(dfdict1['cosine'][i])
    for i in range(len(dfdict2['filename'])):
        if dfdict2['train'][i]:
            labels3.append(dfdict2['cosine'][i])

    plt.figure(figsize=(12, 7))
    plt.hist(labels3, bins=80, color='black')
    plt.show()
    ###########################################    
    '''

    '''
    dfdict = pd.DataFrame.to_dict(pd.read_csv('../data/label_blur_defocus/label/data_label.csv'))
    labels2 = []
    for i in range(len(dfdict['filename'])):
        if dfdict['train'][i]:
            labels2.append(dfdict['cosine'][i])

    plt.figure(figsize=(12, 7))
    plt.hist(labels2, bins=40, color='black')
    plt.show()
    '''
