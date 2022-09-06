import os
import sys
import pdb
from facenet_pytorch import InceptionResnetV1
import torch.optim.lr_scheduler

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from dataset.dataset import FaceDataset
from dataset.dataset2 import FaceDataset2, FaceDataset_val
from models.mobilenet import FaceMobileNetV1, FaceMobileNetV2
from models.resnet import *
import torchvision.transforms as transforms
import pytorch_model_summary
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import torchvision


def visualize(model, input_size, device, iteration, epoch):
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
                for i in range(1, 101):
                    try:
                        blurred_img_fix = cv2.resize(cv2.imread(os.path.join(sample_path, f'fix_{i}.png')),
                                                     (input_size, input_size), interpolation=cv2.INTER_AREA) / 255
                        blurred_img_random = cv2.resize(cv2.imread(os.path.join(sample_path, f'random_{i}.png')),
                                                        (input_size, input_size), interpolation=cv2.INTER_AREA) / 255

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

        plt.figure(figsize=(24, 7))
        plt.subplot(1, 2, 1)
        plt.plot(cos_mean_fix, 'k', linewidth=2, label="Estimated(Fix $\\theta$)")
        plt.plot(real_mean_fix, 'k--', linewidth=2, label="Real(Fix $\\theta$)")
        plt.legend(fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(cos_mean_random, 'k', linewidth=2, label="Estimated(Random $\\theta$)")
        plt.plot(real_mean_random, 'k--', linewidth=2, label="Real(Random $\\theta$)")
        plt.legend(fontsize=15)
        plt.savefig(f"graph_{epoch}_{iteration}.png")

if __name__ == '__main__':
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Your current device is :", device)
    batch = 32
    learning_rate = 3e-4
    input_size = 112
    epochs = 30

    # Getting dataset
    print("Getting dataset ...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)])
    dataset = FaceDataset2('../data/label_blur_defocus/label/data_label.csv', 'cosine', transform, input_size, 'rgb')
    val_dataset = FaceDataset_val('../data/label_val.csv', 'cosine', transform, input_size=input_size, cmap='rgb')
    # Check number of each dataset size
    print(f"Training dataset size : {len(dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    # Dataloaders
    train_dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    iter_length = len(train_dataloader)

    # Instantiate model configuration
    '''
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    '''
    model = torch.load('/data/faceblur/BlurFaceDetection/FaceBlurring/train/checkpoint/resnet18_112/checkpoint_4.pt')
    model.to(device)
    print("Model configuration : ")
    print(
        pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size).to(device), show_input=True))
    # Criterion, Optimizer, Loss history tracker
    criterion = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.3)
    # Create directory to save checkpoints
    os.makedirs("./checkpoint/resnet18_112/", exist_ok=True)

    # Train model
    print("Training ... ")
    for epoch in range(5, epochs):
        training_loss = 0.0
        for i, (image, label) in tqdm(enumerate(train_dataloader)):
            model.train()
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            prediction = model(image)
            loss = criterion(prediction, label.view(-1, 1))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch #{epoch + 1} [{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss / (i + 1):.6f}")
            if (i + 1) % 600 == 0:
                visualize(model, input_size, device, i, epoch)

        scheduler.step()
        torch.save(model, f"./checkpoint/resnet18_112/checkpoint_{epoch}.pt")
