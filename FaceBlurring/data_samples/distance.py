from tqdm import tqdm
import cv2, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

def visualize(model, device, iteration, epoch):
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
                for i in tqdm(range(1, 101)):
                    blurred_img_fix = cv2.imread(os.path.join(sample_path, f'fix_{i}.png')) / 255
                    blurred_img_random = cv2.imread(os.path.join(sample_path, f'random_{i}.png')) / 255
                    try:
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

        plt.figure(figsize=(24, 14))
        plt.plot(cos_mean_fix, color='red', linestyle='dashed', marker='x', markersize=9, linewidth=2, label="Estimated i-Similarity (Fix $\\theta$)")
        plt.plot(real_mean_fix, color='green', linestyle='dashed', marker='o', markersize=9, linewidth=2, label="Real i-Similarity (Fix $\\theta$)")
        plt.plot(cos_mean_random, color='blue', linestyle='dashed', marker='x', markersize=9, linewidth=2, label="Estimated i-Similarity (Random $\\theta$)")
        plt.plot(real_mean_random, color='sienna', linestyle='dashed', marker='o', markersize=9, linewidth=2, label="Real i-Similarity (Random $\\theta$)")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=25)
        plt.legend(fontsize=25)
        plt.savefig(f"graph_{epoch}_{iteration}.png")


if __name__ == '__main__':
    device = 'cuda'
    model = torch.load('/data/faceblur/BlurFaceDetection/FaceBlurring/train/checkpoint/resnet18_112/checkpoint_25.pt')
    visualize(model, device, 10, 10)
