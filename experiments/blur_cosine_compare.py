import os
import cv2
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
from facenet_pytorch import InceptionResnetV1


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    return PSNR


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
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


def cos_sim(A, B):
    # A, B should be feature vector from face recognition
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))



if __name__ == '__main__':
    print("Please check data path & save path")
    ref = cv2.imread('./data/FFHQ_1024/clean/00595/random_0.png')
    image_sample1 = cv2.imread('./data/FFHQ_1024/clean/00595/random_78.png')
    image_sample2 = cv2.imread('./data/FFHQ_1024/clean/00595/random_97.png')
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    ssim1 = ssim(ref, image_sample1)
    ssim2 = ssim(ref, image_sample2)
    psnr1 = psnr(ref, image_sample1)
    psnr2 = psnr(ref, image_sample2)

    emb_clean = resnet(torch.Tensor(ref).permute(2, 0, 1).unsqueeze(0))
    emb_blur1 = resnet(torch.Tensor(image_sample1).permute(2, 0, 1).unsqueeze(0))
    emb_blur2 = resnet(torch.Tensor(image_sample2).permute(2, 0, 1).unsqueeze(0))
    cossim1 = cos_sim(emb_clean.squeeze(0).detach().numpy(), emb_blur1.squeeze(0).detach().numpy())
    cossim2 = cos_sim(emb_clean.squeeze(0).detach().numpy(), emb_blur2.squeeze(0).detach().numpy())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Clean image')
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image_sample1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Cosine similarity : {cossim1:.2f}", fontsize=15)

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(image_sample2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Cosine similarity : {cossim2:.2f}", fontsize=15)

    plt.savefig('results/distance_test/blur_cosine_compare.png')
