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
# from models.mobilenet import FaceMobileNetV1, FaceMobileNetV2
from models.resnet import *
import torchvision.transforms as transforms
import pytorch_model_summary
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import torchvision
import torch.nn as nn

if __name__ == '__main__':
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Your current device is :", device)
    batch = 64
    learning_rate = 1e-3
    input_size = 112
    epochs = 50

     # Getting dataset
    print("Getting dataset ...")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ColorJitter(brightness=0.8)])
    label_list = ['./BlurFaceDetection/FaceBlurring/data/label_deblurGAN/label/data_label.csv',
                 './BlurFaceDetection/FaceBlurring/data/label_defocus/label/data_label.csv',
                 './BlurFaceDetection/FaceBlurring/data/label_random/label/data_label.csv']
    dataset = FaceDataset(label_list ,'cosine', transform, input_size, 'rgb', 'cls')
    
    # Splitting dataset
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    iter_length = len(train_dataloader)

    # Instantiate model configuration
    
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 20)
    model.to(device)

    print("Model configuration : ")
    print(
        pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size).to(device), show_input=True))
    # Criterion, Optimizer, Loss history tracker
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Create directory to save checkpoints
    os.makedirs("./checkpoint/resnet18_112_cls20/", exist_ok=True)

    print("Current CPU random seed :", torch.initial_seed())
    print("Current CUDA random seed :", torch.cuda.initial_seed())

    # Train!
    hist= {'train_loss' : [],
           'val_loss' : [],
           'train_acc' : [],
           'val_acc' : []}

    print("Training ... ")
    for epoch in range(epochs):
        training_loss = 0.0
        total, correct = 0, 0
        for i, (image, label) in tqdm(enumerate(train_dataloader)):
            model.train()
            cls_label, reg_label = label
            image, cls_label, reg_label = image.to(device), cls_label.to(device), reg_label.to(device)
            optimizer.zero_grad()
            prediction = model(image)
            _, predicted = torch.max(prediction.data, 1)
            total += cls_label.size(0)
            correct += (predicted == cls_label).sum().item()
            value = torch.argmax(prediction, dim=1)*0.05
            loss = 0.5*criterion1(prediction, cls_label) + 0.5*criterion2(value, reg_label.view(-1, 1))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch #{epoch + 1} [{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss / (i + 1):.6f} >>>> Training Accuracy : {100*(correct/total):.6f}%")
        
        hist['train_loss'] += [training_loss/len(train_dataloader)]
        hist['train_acc'] += [100*(correct/total)]

        scheduler.step()
        model.eval()

        with torch.no_grad():
            total, correct, val_loss = 0, 0, 0.0
            for i, (image, label) in tqdm(enumerate(val_dataloader)):
                cls_label, reg_label = label
                image, cls_label, reg_label = image.to(device), cls_label.to(device), reg_label.to(device)
                prediction = model(image)
                _, predicted = torch.max(prediction.data, 1)
                total += cls_label.size(0)
                correct += (predicted == cls_label).sum().item()
                value = torch.argmax(prediction, dim=1)*0.05
                loss = 0.5*criterion1(prediction, cls_label) + 0.5*criterion2(value, reg_label.view(-1, 1))
                val_loss += loss.item()
        
            print(f"Epoch #{epoch + 1} [{i}/{len(val_dataloader)}] >>>> Val loss : {val_loss / len(val_dataloader):.6f} >>>> Val Accuracy : {100*(correct/total):.6f}%")
            
            hist['val_loss'] += [val_loss/len(val_dataloader)]
            hist['val_acc'] += [100*(correct/total)]            
            
            torch.save(model, f"./checkpoint/resnet18_112_cls20/checkpoint_{epoch}.pt")

# Plotting
plt.figure(figsize=(24, 7))
plt.subplot(1, 2, 1)
plt.plot(hist['train_loss'], 'k', linewidth=2, label="Training loss")
plt.plot(hist['val_loss'], 'k--', linewidth=2, label="Validation loss")
plt.legend(fontsize=15)

plt.subplot(1, 2, 2)
plt.plot(hist['train_acc'], 'k', linewidth=2, label="Training Accuracy")
plt.plot(hist['val_acc'], 'k--', linewidth=2, label="Validation Accuracy")
plt.legend(fontsize=15)
plt.savefig(f"./result/graph_loss_resnet_cls20.png")
plt.close()