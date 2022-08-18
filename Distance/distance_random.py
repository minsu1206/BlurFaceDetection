from tqdm import tqdm
import cv2, os
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
import matplotlib.pyplot as plt

# Get average distances from angle random samples
def calculate_distances(img_dir, idx):
    reference_img = cv2.imread(os.path.join(img_dir, 'random_0.png'))
    emb_clean = resnet(torch.Tensor(reference_img).permute(2, 0, 1).unsqueeze(0))
    blurred_img = cv2.imread(os.path.join(img_dir, f'random_{idx}.png'))
    emb_blur = resnet(torch.Tensor(blurred_img).permute(2, 0, 1).unsqueeze(0))
    l1 = L1_distance(emb_clean, emb_blur)
    l2 = L2_distance(emb_clean, emb_blur)
    cossim = cos_sim(emb_clean.squeeze(0).detach().numpy(), emb_blur.squeeze(0).detach().numpy())

    return blurred_img, l1, l2, cossim

def L1_distance(emb1, emb2):
    return torch.abs(torch.sum(emb1 - emb2))

def L2_distance(emb1, emb2):
    return torch.sqrt(torch.sum(torch.square(emb1 - emb2)))

def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

if __name__ == '__main__':
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    path = './data/FFHQ_1024/clean/'

    look_upto = 0
    l1_mean, l2_mean, cos_mean = np.zeros(100), np.zeros(100), np.zeros(100)

    for subpath in os.listdir(path):
        if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
            sample_path = os.path.join(path, subpath)
            look_upto += 1
            for i in tqdm(range(1, 101)):
                _, l1, l2, cossim = calculate_distances(sample_path, i)
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
    plt.savefig("random_L1L2COS.png")
