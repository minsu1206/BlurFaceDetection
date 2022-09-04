import pdb

import torch
import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
class FaceDataset(Dataset):
    def __init__(self, data_root, option='blur', method='defocus',
                 transform=None, input_size=None, check=False):
        '''
            Dataset
            Check csv file(be sure create label first)
            data_root : Top directory of dataset
            method : blur method
            input_size : image input size
            check : get all images from lower directory(boolean)
        '''

        assert method in ['defocus', 'deblurGAN', 'random'], "Not available method"
        self.transform = transform
        self.method = method
        self.data_root = data_root
        self.option = option
        self.label_path = f'../data/label_blur_{method}/label/data_label.csv'

        if check:
            if option == 'clean':
                self.sample_paths = self._get_clean_samples()
                self.labels = np.zeros(len(self.sample_paths))

            elif option == 'blur':
                self.sample_paths, self.labels = self._get_blur_samples()

            elif option == 'both':
                self.blur_paths, self.blur_labels = self._get_blur_samples()
                self.clean_paths = self._get_clean_samples()
                self.clean_labels = np.zeros(len(self.sample_paths))

            else:
                raise ValueError("option should be 'clean' or 'blur' or 'both'.")
        else:
            if os.path.isfile(self.label_path):
                if option == 'clean':
                    df = pd.read_csv(self.label_path)
                    self.sample_paths = df['filename'].replace('blur_defocus', 'clean', regex=True)
                    self.labels = np.zeros(len(self.sample_paths))

                elif option == 'blur':
                    df = pd.read_csv(self.label_path)
                    self.sample_paths = df['filename']
                    self.labels = df['cosine']

                elif option == 'both':
                    df = pd.read_csv(self.label_path)
                    self.clean_paths = df['filename'].replace('blur_defocus', 'clean', regex=True)
                    self.clean_labels = np.zeros(len(self.clean_paths))
                    self.blur_paths = df['filename']
                    self.blur_labels = df['cosine']
                    pdb.set_trace()

            else:
                raise ValueError("Create label first(run create_blur_image.py first)")

        if input_size is None:
            self.input_size = 1024
        else:
            self.input_size = input_size

    def _get_clean_samples(self):
        paths = []
        roots = self.data_root

        for root in roots:
            for (path, directory, files) in os.walk(root):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext in ['.png', '.jpg', 'PNG', 'JPG', 'JPEG'] and 'clean' in path:
                        paths += [os.path.join(path, filename)]
        return paths

    def _get_blur_samples(self):
        paths = []
        labels = []
        label_path = self.label_path
        roots = self.data_root
        assert os.path.isfile(label_path), "label file does not exist"
        df = pd.read_csv(label_path)
        assert self.calc in list(df.columns.values), 'Regenerate label with same metric'

        for root in roots:
            for (path, directory, files) in os.walk(root):
                for filename in files:
                    ext = os.path.splitext(filename)[-1]
                    if ext in ['.png', '.jpg', 'PNG', 'JPG', 'JPEG'] and 'blur_' + self.method in path:
                        filepath = os.path.join(path, filename)
                        paths += [filepath]
                        labels.append(np.float32(df.loc[df['filename'] == filepath][self.calc].item()))

        return paths, labels

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        if self.option != 'both':
            img_path, label = self.sample_paths[idx], self.labels[idx]
            try:
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            except:
                print(img_path, ': corrupted')
                img_path, label = self.sample_paths[idx + 1], self.labels[idx + 1]
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            if self.transform:
                image = self.transform(image).float()

            return image, torch.from_numpy(np.asarray(label))

        else:
            clean_pth, clean_lb = self.clean_paths[idx], self.clean_labels[idx]
            blur_pth, blur_lb = self.blur_paths[idx], self.blur_labels[idx]
            try:
                clean = cv2.cvtColor(cv2.imread(clean_pth), cv2.COLOR_BGR2RGB)
                blur = cv2.cvtColor(cv2.imread(blur_pth), cv2.COLOR_BGR2RGB)

            except:
                clean_pth, clean_lb = self.clean_paths[idx+1], self.clean_labels[idx+1]
                blur_pth, blur_lb = self.blur_paths[idx+1], self.blur_labels[idx+1]
                clean = cv2.cvtColor(cv2.imread(clean_pth), cv2.COLOR_BGR2RGB)
                blur = cv2.cvtColor(cv2.imread(blur_pth), cv2.COLOR_BGR2RGB)

            clean = cv2.resize(clean, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            blur = cv2.resize(blur, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

            if self.transform:
                clean = self.transform(clean).float()
                blur = self.transform(blur).float()

            image = {'clean' : clean, 'blur' : blur}
            label = {'clean': torch.from_numpy(np.asarray(clean_lb)),
                     'blur': torch.from_numpy(np.asarray(blur_lb))}

            return image, label
