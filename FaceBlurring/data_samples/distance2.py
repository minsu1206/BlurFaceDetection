from tqdm import tqdm
import cv2, os
import torch
import numpy as np
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt
import pickle
import pdb

def calculate_distances(img_dir, idx, option):
    reference_img = cv2.imread(os.path.join(img_dir, f'{option}_0.png'))
    emb_clean = resnet(torch.Tensor(reference_img).permute(2, 0, 1).unsqueeze(0).cuda())
    blurred_img = cv2.imread(os.path.join(img_dir, f'{option}_{idx}.png'))
    emb_blur = resnet(torch.Tensor(blurred_img).permute(2, 0, 1).unsqueeze(0).cuda())
    cossim = F.cosine_similarity(emb_clean, emb_blur, 1).item()
    l1 = torch.norm(emb_clean-emb_blur, p=1, dim=0).mean().item()
    l2 = torch.norm(emb_clean-emb_blur, p=2, dim=0).mean().item()
    return 1-cossim, l1, l2

if __name__ == '__main__':
    '''
    resnet = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
    path = './data/FFHQ_1024/clean/'

    look_upto = 0
    cos_mean = np.zeros(100)
    l1_mean, l2_mean = np.zeros(100), np.zeros(100)

    cos_mean2 = np.zeros(100)
    l1_mean2, l2_mean2 = np.zeros(100), np.zeros(100)

    for subpath in os.listdir(path):
        if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
            sample_path = os.path.join(path, subpath)
            look_upto += 1
            for i in tqdm(range(100, 0, -1)):
                icos, l1, l2 = calculate_distances(sample_path, i, 'random')
                icos2, l12, l22 = calculate_distances(sample_path, i, 'fix')
                cos_mean[i-1] += icos
                l1_mean[i-1] += l1
                l2_mean[i-1] += l2

                cos_mean2[i - 1] += icos2
                l1_mean2[i - 1] += l12
                l2_mean2[i - 1] += l22

            if look_upto == 30:
                break
    cos_mean /= look_upto
    l1_mean /= look_upto
    l2_mean /= look_upto
    cos_mean2 /= look_upto
    l1_mean2 /= look_upto
    l2_mean2 /= look_upto

    
    save_info = {'random_cos' : cos_mean, 'random_l1' : l1_mean, 'random_l2' : l2_mean,
                 'fix_cos' : cos_mean2, 'fix_l1' : l1_mean2, 'fix_l2' : l2_mean2}
    with open('data.pickle', 'wb') as f:
        pickle.dump(save_info, f, pickle.HIGHEST_PROTOCOL)
    '''
    with open('data.pickle','rb') as f:
        save_info = pickle.load(f)

    l2_mean, l1_mean, cos_mean, l2_mean2, l1_mean2, cos_mean2 = save_info['random_l2'], save_info['random_l1'], save_info['random_cos'], save_info['fix_l2'], save_info['fix_l1'], save_info['fix_cos']

    fig, ax1 = plt.subplots(figsize=(24, 14))
    line1 = ax1.plot(l2_mean, color='green', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                     label="Average L2 distance (Random, $\\theta$)")
    line2 = ax1.plot(cos_mean, color='blue', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                     label="Average i-Similarity (Random $\\theta$)")

    ax2 = ax1.twinx()
    line3 = ax2.plot(l1_mean, color='red', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                     label="Average L1 distance (Random $\\theta$)")

    lines = line3 + line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel('L2 distance & i-Similarity',labelpad=25, fontsize=25)
    ax1.set_ylim([-0.15, 2.3])
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel('L1 distance', labelpad=20, fontsize=25)
    ax2.legend(lines, labs, fontsize=25)
    plt.savefig(f"graph_distance_random.png")

    #################################################################
    fig, ax1 = plt.subplots(figsize=(24, 14))
    line1 = ax1.plot(l2_mean2, color='green', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                label="Average L2 distance (Fix $\\theta$)")
    line2 = ax1.plot(cos_mean2, color='blue', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                label="Average i-Similarity (Fix $\\theta$)")

    ax2 = ax1.twinx()
    line3 = ax2.plot(l1_mean2, color='red', linestyle='dashed', marker='x', markersize=8, linewidth=2,
                label="Average L1 distance (Fix $\\theta$)")

    lines = line3 + line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_ylabel('L2 distance & i-Similarity', labelpad=25, fontsize=25)

    ax1.set_ylim([-0.15, 2.3])
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel('L1 distance', labelpad=20, fontsize=25)
    ax2.legend(lines, labs, fontsize=25)
    plt.savefig(f"graph_distance_fix.png")
