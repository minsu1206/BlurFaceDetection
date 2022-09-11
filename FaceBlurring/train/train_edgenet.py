import os
import sys
import torch.optim.lr_scheduler
import models.model
import pdb
import torch.nn as nn
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from timm.models import create_model
from torch.utils.data import DataLoader, random_split
from dataset.dataset import FaceDataset
import torchvision.transforms as transforms
import pytorch_model_summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pickle5 as pickle

def visualize(model, input_size, device):
    model.eval()
    with torch.no_grad():
        path = '../data_samples/samples'
        cos_mean_random = np.zeros(100)
        cos_mean_fix = np.zeros(100)

        with open('../data_samples/random_reference.pkl', 'rb') as f:
            real_mean_random = pickle.load(f)

        with open('../data_samples/fix_reference.pkl', 'rb') as f:
            real_mean_fix = pickle.load(f)

        iter = 0
        for subpath in tqdm(os.listdir(path)):
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
        plt.savefig(f"graph_edgenet_simple_visualize.png")

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
                                    transforms.ColorJitter(brightness=0.8)], )
    dataset = FaceDataset("../", 'blur', 'all', transform)
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size -train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #val_dataset = FaceDataset_val('../data/label_val.csv', 'cosine', transform, input_size=input_size, cmap='rgb')
    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    iter_length = len(train_dataloader)
    model = create_model(
        'edgenext_xx_small_bn_hs',
        pretrained=True,
        num_classes=1,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        input_res=input_size,
        classifier_dropout=0.0,
    )
    model.head = nn.Sequential(
        nn.Linear(168, 1),
        nn.Sigmoid()
    )
    model.to(device)
    print("Model configuration : ")
    print(
        pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size).to(device), show_input=True))
    # Criterion, Optimizer, Loss history tracker
    criterion = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # Create directory to save checkpoints
    os.makedirs("./checkpoint/edgenet_simple/", exist_ok=True)

    print("Current CPU random seed :", torch.initial_seed())
    print("Current CUDA random seed :", torch.cuda.initial_seed())

    # Train model
    hist = {'train_loss' : [], 'val_loss' : []}
    print("Training ... ")
    for epoch in range(epochs):
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
                    f"Epoch #[{epoch+1}/{epochs}] [{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss / (i + 1):.6f}")
        scheduler.step()
        torch.save(model, f"./checkpoint/edgenet_simple/checkpoint_{epoch}.pt")
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, (image, label) in tqdm(enumerate(val_dataloader)):
                model.train()
                image, label = image.to(device), label.to(device)
                prediction = model(image)
                loss = criterion(prediction, label.view(-1, 1))
                validation_loss += loss.item()
                
            print(f"Epoch #[{epoch+1}/{epochs}] [{i}/{len(val_dataloader)}] >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}")
            hist['train_loss'].append(training_loss/len(train_dataloader))
            hist['val_loss'].append(validation_loss/len(val_dataloader))
            
    
    plt.figure(figsize=(12, 7))
    plt.title("Loss history", fontsize=20)
    plt.plot(hist['train_loss'], 'k', linewidth=2, label="Training loss per epoch")
    plt.plot(hist['val_loss'], 'k--', linewidth=2, label="Validation loss per epoch")
    plt.legend(fontsize=15)
    plt.savefig("edgenet_simple_loss.png")


    model = torch.load("./checkpoint/edgenet_simple/checkpoint_49.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    visualize(model, 112, "cuda" if torch.cuda.is_available() else "cpu")