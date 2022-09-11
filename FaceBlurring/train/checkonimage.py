import torch
import cv2
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = torch.load("./checkpoint/resnet18_complex/checkpoint_49.pt")
    model.to(device)
    model.eval()
    
    '''
    root = '../data_samples/samples/28161'
    filename_list = sorted(os.listdir(root), key=lambda x : int(os.path.splitext(x)[0].split('_')[-1]))
    value = []
    
    for filename in tqdm(filename_list):
        if 'random' in filename:
            image = cv2.resize(cv2.imread(os.path.join(root, filename)), (112, 112), interpolation=cv2.INTER_AREA) / 255
            image_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(image_tensor)
            value += [output.item()]
    '''
    image1 = cv2.resize(cv2.cvtColor(cv2.imread('../data_samples/samples/28026/fix_0.png'), cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA) / 255
    image2 = cv2.resize(cv2.cvtColor(cv2.imread('../data_samples/samples/28026/random_30.png'), cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA) / 255
    image3 = cv2.resize(cv2.cvtColor(cv2.imread('../data_samples/samples/28026/fix_60.png'), cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA) / 255
    image4 = cv2.resize(cv2.cvtColor(cv2.imread('../data_samples/samples/28026/random_90.png'), cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA) / 255
    
    tensor1 = torch.Tensor(image1).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor2 = torch.Tensor(image2).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor3 = torch.Tensor(image3).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor4 = torch.Tensor(image4).permute(2, 0, 1).unsqueeze(0).to(device)

    output1 = model(tensor1)
    output2 = model(tensor2)
    output3 = model(tensor3)
    output4 = model(tensor4)
                        
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image1)
    plt.title(f"{output1.item():.6f}", fontsize=25)
    plt.subplot(1, 4, 2)
    plt.imshow(image2)
    plt.title(f"{output2.item():.6f}", fontsize=25)
    plt.subplot(1, 4, 3)
    plt.imshow(image3)
    plt.title(f"{output3.item():.6f}", fontsize=25)
    plt.subplot(1, 4, 4)
    plt.imshow(image4)
    plt.title(f"{output4.item():.6f}", fontsize=25)
    plt.savefig("validate3.png")