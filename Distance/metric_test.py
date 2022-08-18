import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def random_calculate_metrics(img_dir, idx):
    reference_img = cv2.imread(os.path.join(img_dir, 'random_0.png'))
    blurred_img = cv2.imread(os.path.join(img_dir, f'random_{idx}.png'))

    return psnr(reference_img, blurred_img), ssim(reference_img, blurred_img)

def fix_calculate_metrics(img_dir, idx):
    reference_img = cv2.imread(os.path.join(img_dir, 'fix_0.png'))
    blurred_img = cv2.imread(os.path.join(img_dir, f'fix_{idx}.png'))

    return psnr(reference_img, blurred_img), ssim(reference_img, blurred_img)

if __name__ == '__main__':
    path = './data/FFHQ_1024/clean/'

    look_upto = 0
    psnr_mean, ssim_mean = np.zeros(100), np.zeros(100)

    for subpath in os.listdir(path):
        if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
            sample_path = os.path.join(path, subpath)
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
    plt.savefig("fix_psnr.png")
    plt.legend()

    plt.figure(figsize=(12, 7))
    plt.plot(ssim_mean, 'k-', linewidth=2)
    plt.title("Average SSIM", fontsize=20)
    plt.savefig("fix_ssim.png")