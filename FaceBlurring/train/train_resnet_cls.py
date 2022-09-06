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

if __name__ == '__main__':
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Your current device is :", device)
    batch = 16
    learning_rate = 1e-4
    input_size = 112
    epochs = 30

    # Getting dataset
    print("Getting dataset ...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)])
    dataset = FaceDataset2('/data/faceblur/BlurFaceDetection/FaceBlurring/data/label_blur_defocus/label/data_label.csv', 'cosine', transform, input_size, 'rgb', 'cls')
    val_dataset = FaceDataset_val('/data/faceblur/BlurFaceDetection/FaceBlurring/data/label_val.csv', 'cosine', transform, input_size=input_size, cmap='rgb')
    # Check number of each dataset size
    print(f"Training dataset size : {len(dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")

    # Dataloaders
    train_dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    iter_length = len(train_dataloader)

    # Instantiate model configuration
    
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 626)
    model.to(device)
    print("Model configuration : ")
    print(
        pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size).to(device), show_input=True))
    # Criterion, Optimizer, Loss history tracker
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.3)
    # Create directory to save checkpoints
    os.makedirs("./checkpoint/resnet18_112_cls/", exist_ok=True)

    # Train model
    print("Training ... ")
    for epoch in range(epochs):
        training_loss = 0.0
        for i, (image, label) in tqdm(enumerate(train_dataloader)):
            model.train()
            cls_label, reg_label = label
            image, cls_label, reg_label = image.to(device), cls_label.to(device), reg_label.to(device)
            optimizer.zero_grad()
            prediction = model(image)
            value = torch.argmax(prediction, dim=1)*0.001
            loss = 0.5*criterion1(prediction, cls_label) + 0.5*criterion2(value, reg_label.view(-1, 1))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print(
                    f"Epoch #{epoch + 1} [{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss / (i + 1):.6f}")

        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, (image, label) in tqdm(enumerate(val_dataloader)):
                image, label = image.to(device), label.to(device)
                prediction = model(image)
                
                loss = criterion(prediction, label)
                validation_loss += loss.item()
                
            print(f"(Val)Epoch #{epoch + 1} [{i}/{len(train_dataloader)}] >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}")
        scheduler.step()
        torch.save(model, f"./checkpoint/resnet18_112_cls/checkpoint_{epoch}.pt")