import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import path
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import InceptionResnetV1

def calculate_distances(img_dir, idx, mode, model, device):

    reference_img = cv2.imread(os.path.join(img_dir, f'{mode}_0.png'))
    reference_tensor = torch.Tensor(reference_img).permute(2, 0, 1).unsqueeze(0).to(device)
    blurred_img = cv2.imread(os.path.join(img_dir, f'{mode}_{idx}.png'))
    blurred_tensor = torch.Tensor(blurred_img).permute(2, 0, 1).unsqueeze(0).to(device)

    emb_clean = model(refernce_tensor)
    emb_blur = model(blurred_tensor)

    l1 = L1_distance(emb_clean, emb_blur)
    l2 = L2_distance(emb_clean, emb_blur)

    emb_clean = emb_clean.to('cpu')
    emb_blur = emb_blur.to('cpu')

    cossim = cos_sim(emb_clean.squeeze(0).detach().numpy(), emb_blur.squeeze(0).detach().numpy())

    return blurred_img, l1, l2, cossim

def calculate_pixel_metric(img_dir, idx, mode, model):
    reference_img = cv2.imread(os.path.join(img_dir, f'{mode}_0.png'))
    blurred_img = cv2.imread(os.path.join(img_dir, f'{mode}_{idx}.png'))
    return psnr(reference_img, blurred_img), ssim(reference_img, blurred_img)

def L1_distance(emb1, emb2):
    return torch.abs(torch.sum(emb1 - emb2))

def L2_distance(emb1, emb2):
    return torch.sqrt(torch.sum(torch.square(emb1 - emb2)))

def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    return PSNR

def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def plot_blur_sample_distance(mode='random', metric='pixel'):
    mode = args.mode
    device = args.device
    metric = args.metric
    
    look_upto = 0

    if metric == 'distance':
        '''Get average distances from angle fix/random samples'''
        resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        path = './data/FFHQ_1024/clean/'
        
        l1_mean, l2_mean, cos_mean = np.zeros(100), np.zeros(100), np.zeros(100)

        for subpath in os.listdir(path):
            if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
                sample_path = os.path.join(path, subpath)
                look_upto += 1
                for i in tqdm(range(1, 101)):
                    _, l1, l2, cossim = calculate_distances(sample_path, i, mode, resnet, device)
                    l1_mean[i-1] += l1
                    l2_mean[i-1] += l2
                    cos_mean[i-1] += cossim

                if look_upto == 30:
                    break

        l1_mean /= look_upto
        l2_mean /= look_upto
        cos_mean /= look_upto

        plt.figure(figsize=(12, 7))
        plt.plot(l1_mean, color='darkred', linewidth=2, label='Average L1 distance')
        plt.plot(l2_mean, color='darkgreen', linewidth=2, label="Average L2 distance")
        plt.plot(cos_mean, color='darkblue', linewidth=2, label='Average Cosine similarity')
        plt.legend(fontsize=15)
        plt.savefig(f"results/distance_test/{mode}_L1L2COS.png")

    if distance == 'pixel':
        # TODO
        psnr_mean, ssim_mean = np.zeros(100), np.zeros(100)

        for subpath in os.listdir(path):
            if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
                sample_path = os.path.join(path,subpath)
                look_upto += 1
                for i in tqdm(range(1, 101)):
                    p, s = fix_calculate_metrics(sample_path, i)
                    psnr_mean[i-1] += p
                    ssim_mean[i-1] += s

                if look_upto == 30:
                    break
        
        psnr_mean /= look_upto
        ssim_mean /= look_upto

        plt.figure(figsize=(12, 7))
        plt.plot(psnr_mean, 'k-', linewidth=2)
        plt.title("Average PSNR", fontsize=20)
        plt.savefig(f"result/distance_test/{mode}_psnr.png")
        plt.legend()

        plt.figure(figsize=(12, 7))
        plt.plot(ssim_mean, 'k-', linewidth=2)
        plt.title("Average SSIM", fontsize=20)
        plt.savefig(f"result/distance_test/{mode}_ssim.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='random', help='Select type of data samples. It can be "fix" or "random"')
    parser.add_argument('--cuda', type=str, default='cpu')
    parser.add_argument('--metric', type=str, default='pixel', help='[pixel, distance]')
    args = parser.parse_args()

    plot_blur_sample_distance(args)