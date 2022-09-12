import os
import argparse
from collections import defaultdict

from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from blur_iterative import iterative_blur_n

from multiprocessing import Pool, cpu_count
from functools import partial

def get_img_path(root_path):
    '''
    get images' path from directory path

    Args:
        root path: Top directory path you want to obtain

    Returns:
        image path list
    '''
    img_path_list = []
    for (root, _, files) in os.walk(root_path):
        root = root.replace('\\', '/')
        if len(files) != 0:
            for img_path in files:
                filename, file_extension = os.path.splitext(img_path)
                if file_extension in ['.jpg', '.png']:
                    img_path_list.append(root + '/' + img_path)
    return img_path_list

def save_iterative_blur_img(root_path, n, model, device):
    '''
    obtain blur images by applying the blur method several times
    and save iterative blur images and cossim

    Args:
        root_path: Top directory path you want to obtain
        n: iterative number
    
    '''
    dsize=(112, 112)
    dict_for_label = {'file_name' : [], '1-cossim' : []}

    for (root, mid_list, files) in os.walk(root_path):
        root = root.replace('\\', '/')
        for mid in mid_list:
            clean_save_path = f'../data/{dsize[0]}/clean/{mid}'
            blur_save_path = f'../data/{dsize[0]}/blur/{mid}'
            os.makedirs(clean_save_path, exist_ok=True)
            os.makedirs(blur_save_path, exist_ok=True)

    img_path_list = get_img_path(root_path)
    #################################################
    for img_path in tqdm(img_path_list):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        clean_img, blur_img_list, cossim_list = iterative_blur_n(model, img, n, dsize, 
        blur_method_list=['deblurGAN', 'defocus'], device=device)

    #################################################
        clean_img_path = img_path.replace('sample_root', f'{dsize[0]}')
        cv2.imwrite(clean_img_path, clean_img)

        for idx, blur_img in enumerate(blur_img_list):
            blur_img_path = img_path.replace('sample_root', f'{dsize[0]}')
            blur_img_path = blur_img_path.replace('clean', 'blur')
            base_name = os.path.basename(img_path)
            file_num, _ = os.path.splitext(base_name)
            blur_img_path = blur_img_path.replace(file_num, file_num+'_'+str(idx+1))
            dict_for_label['file_name'].append(blur_img_path)
            dict_for_label['1-cossim'].append(1-cossim_list[idx])
            cv2.imwrite(blur_img_path, blur_img)

    save_dir = ".."+os.path.sep+os.path.join('data', f"{dsize[0]}", 'label')
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(dict_for_label)
    df.to_csv(os.path.join(save_dir, "label.csv"))


def save_iterative_blur_only_img(root_path, n, device):
    '''
    obtain blur images by applying the blur method several times
    and save iterative blur images and cossim

    Args:
        root_path: Top directory path you want to obtain
        n: iterative number
    
    '''
    # dsize=(112, 112)
    dict_for_label = {'file_name' : [], '1-cossim' : []}
    model = None

    # for (root, mid_list, files) in os.walk(root_path):
    #     root = root.replace('\\', '/')
    #     for mid in mid_list:
    #         clean_save_path = f'../data/{dsize[0]}/clean/{mid}'
    #         blur_save_path = f'../data/{dsize[0]}/blur/{mid}'
    #         os.makedirs(clean_save_path, exist_ok=True)
    #         os.makedirs(blur_save_path, exist_ok=True)

    os.makedirs(root_path.replace('clean', 'blur'), exist_ok=True)
    img_path_list = get_img_path(root_path)
    #################################################
    for img_path in tqdm(img_path_list):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        clean_img, blur_img_list, _ = iterative_blur_n(model, img, n, dsize=None, 
        blur_method_list=['deblurGAN', 'defocus'], device=device)
    #################################################
        # clean_img_path = img_path.replace('sample_root', f'{dsize[0]}')
        # cv2.imwrite(clean_img_path, clean_img)

        for idx, blur_img in enumerate(blur_img_list):
            # blur_img_path = img_path.replace('sample_root', f'{dsize[0]}')
            # blur_img_path = blur_img_path.replace('clean', 'blur')
            # base_name = os.path.basename(img_path)
            # file_num, _ = os.path.splitext(base_name)
            # blur_img_path = blur_img_path.replace(file_num, file_num+'_'+str(idx+1))
            
            # [9/6 - KMS]
            blur_img_path = img_path.replace('clean', 'blur').replace('.png', f'_{idx}.png')
            cv2.imwrite(blur_img_path, blur_img)

    # save_dir = ".."+os.path.sep+os.path.join('data', f"{dsize[0]}", 'label')
    # os.makedirs(save_dir, exist_ok=True)


def save_iterative_blur_only_img_multi(root_path, n, device):
    '''
    obtain blur images by applying the blur method several times
    and save iterative blur images and cossim

    Args:
        root_path: Top directory path you want to obtain
        n: iterative number
    
    '''
    # dsize=(112, 112)
    num_workers = cpu_count()
    dict_for_label = {'file_name' : [], '1-cossim' : []}
    model = None

    pool = Pool(processes=num_workers)
    os.makedirs(root_path.replace('clean', 'blur'), exist_ok=True)
    img_path_list = get_img_path(root_path)

    works = partial(work_single, model, n)
    with pool as p:
        _ = list(tqdm(
            p.imap_unordered(works, img_path_list), total=len(img_path_list)))

def work_single(model, n, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    clean_img, blur_img_list, _ = iterative_blur_n(model, img, n, dsize=None, 
    blur_method_list=['deblurGAN', 'defocus'], device='cpu')
    for idx, blur_img in enumerate(blur_img_list):
        blur_img_path = img_path.replace('clean', 'blur').replace('.png', f'_{idx}.png')
        # print(blur_img_path)
        cv2.imwrite(blur_img_path, blur_img)



def draw_distribution(label_path):
    '''
    Draw x-axis: number of blur methods applied and y-axis: [1-cosine similarity]

    Args:
        label_path: label path containing blur label
    '''
    df = pd.read_csv(label_path)
    cossim_dict = defaultdict(list)
    cossim_list = []

    for file_path, sim_label in zip(df['file_name'], df['1-cossim']):
        key = int(file_path.split('_')[1][:-4]) # blur count
        cossim_dict[key].append(sim_label)

    for key in cossim_dict:
        cossim_list.append(np.mean(np.array(cossim_dict[key])))
    plt.plot(np.arange(1, len(cossim_list)+1), cossim_list)
    plt.xlabel('iterative count')
    plt.ylabel('1-cossim')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program creates iterative blur images.')
    parser.add_argument('--path', type=str, default='../data/FFHQ_512/clean')
    parser.add_argument('--n', type=int, default=4, help='Maximum number of times to repeat blur')
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--wo', action='store_true')
    args = parser.parse_args()

    # [9/3] 수정

    # [9/3] Test : Batch Forwarding?
    # dummy = torch.zeros((8, 3, 256, 256)).to(device)
    # dummy_out = model(dummy)
    # print(dummy_out.shape)    >> [8, 512]
    
    # [9/6] 수정
    # 
    if args.wo:
        
        # no resize
        if args.multi:
            save_iterative_blur_only_img_multi(args.path, args.n, device='cpu')
        else:
            save_iterative_blur_only_img(args.path, args.n, device=device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        save_iterative_blur_img(args.path, args.n, model=model, device=device)

