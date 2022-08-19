import os
import sys
import math
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from utils import ssim, psnr
from dataset import FaceDataset
from embedding.embedding import L1_distance, L2_distance, cos_sim

def get_data_distribution(config_file_path = "../config/test.txt"):
    '''
    Show distribution of blur/sharp dataset
    metric : [PSNR, SSIM, L1 distance, L2 distance, cosine similarity, same/different person]

        Args : 
            config_file_path (str)

        Returns : 
            None
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_dataset_blur = FaceDataset(config_file_path, 'blur')
    face_dataset_clean = FaceDataset(config_file_path, 'clean')

    L1_list, L2_list, cossim_list = [], [], []
    PSNR_list, SSIM_list = [], []
    same_list = [0, 0] # Save 1 if cosine similarity > cossim_threshold(consider as a same person)
    cossim_threshold = 0.4

    resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

    for clean, blur in tqdm(zip(face_dataset_clean, face_dataset_blur)):
        clean_img = clean[0]
        blur_img = blur[0]
        blur_degree = blur[1]

        PSNR_list.append(blur_degree)
        SSIM_list.append(ssim(clean_img, blur_img))

        clean_img_tensor = torch.from_numpy(clean_img).to(device).permute(2, 0, 1).unsqueeze(0).float()
        blur_img_tensor = torch.from_numpy(blur_img).to(device).permute(2, 0, 1).unsqueeze(0).float()
        clean_img_embed = resnet(clean_img_tensor).cpu().squeeze(0).detach()
        blur_img_embed = resnet(blur_img_tensor).cpu().squeeze(0).detach()

        L1_list.append(L1_distance(clean_img_embed, blur_img_embed))
        L2_list.append(L2_distance(clean_img_embed, blur_img_embed))
        cossim = cos_sim(clean_img_embed, blur_img_embed)
        cossim_list.append(cossim)
        same_list[1 if cossim > cossim_threshold else 0] += 1

    plt.figure(figsize=(24, 30))
    plt.subplot(3, 2, 1)
    plt.hist(PSNR_list, 30, range=[25, 40])
    plt.xlabel('PSNR')
    plt.ylabel('count')
    plt.title('PSNR Distribution')
    plt.subplot(3, 2, 2)
    plt.hist(SSIM_list, 30)
    plt.xlabel('SSIM')
    plt.ylabel('count')
    plt.title('SSIM Distribution')
    plt.subplot(3, 2, 3)
    plt.hist(L1_list, 30)
    plt.xlabel('L1 distance')
    plt.ylabel('count')
    plt.title('L1 Distance Distribution')
    plt.subplot(3, 2, 4)
    plt.hist(L2_list, 30)
    plt.xlabel('L2 distance')
    plt.ylabel('count')
    plt.title('L2 Distance Distribution')
    plt.subplot(3, 2, 5)
    plt.hist(cossim_list, 30)
    plt.xlabel('cosine similarity')
    plt.ylabel('count')
    plt.title('Cosine Similarity Distribution')
    plt.subplot(3, 2, 6)
    plt.bar(['Different', 'Same'], same_list)
    plt.ylabel('count')
    plt.title('Recognition Distribution')
    plt.show()

get_data_distribution(config_file_path = "../config/test.txt")