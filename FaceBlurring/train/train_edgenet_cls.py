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
from dataset.dataset2 import FaceDataset2, FaceDataset_val
import torchvision.transforms as transforms
import pytorch_model_summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pickle5 as pickle

def visualize_cls(model, input_size, device, cls_num):
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
                        _, predicted_fix = torch.max(estimated_fix.data, 1)
                        
                        blurred_img_random = torch.Tensor(blurred_img_random).permute(2, 0, 1).unsqueeze(0).to(device)
                        estimated_random = model(blurred_img_random)
                        _, predicted_random = torch.max(estimated_random.data, 1)
                        
                        estimated_random = predicted_random*(1/cls_num)
                        estimated_fix = predicted_fix*(1/cls_num)

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
        plt.savefig(f"./result/graph_edgenet_cls_20_visualize.png")
        plt.close()

class WeightedMSELoss(nn.Module):
    def __init__(self, num_classes=20):
        super(WeightedMSELoss, self).__init__()
        self.c1 = nn.CrossEntropyLoss().to("cuda" if torch.cuda.is_available else 'cpu')
        self.cls = torch.arange(1/(num_classes*2), 1, 1/num_classes)
        
    def forward(self, prob, cls_label, reg_label):
        self.cls_rep = self.cls.repeat(prob.size(0), 1).to("cuda" if torch.cuda.is_available else 'cpu')
        loss1 = self.c1(prob, cls_label) # cross entropy loss
        diff_tensor = torch.sqrt(torch.square(reg_label-self.cls_rep))
        loss2 = torch.sum(diff_tensor*torch.sigmoid(prob)) # probability based mse error
        
        return loss1 + 0.01*loss2
        

if __name__ == '__main__':
    # Hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Your current device is :", device)
    batch = 64
    learning_rate = 1e-3
    input_size = 112
    epochs = 50
    n_classes = 10

    # Getting dataset
    print("Getting dataset ...")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ColorJitter(brightness=0.8)], )
    label_list = ['../data/label_random/label/data_label.csv', '../data/label_defocus/label/data_label.csv', '../data/label_deblurGAN/label/data_label.csv']
    dataset = FaceDataset2(label_list, 'cosine', transform, input_size, 'rgb', 'cls', num_classes=n_classes)
    dataset_size = len(dataset)
    train_size = int(dataset_size*0.8)
    val_size = dataset_size -train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
#     val_dataset = FaceDataset_val('../data/label_val.csv', 'cosine', transform, input_size=input_size, cmap='rgb', option='cls', num_classes=n_classes)
    # Check number of each dataset size
    print(f"Training dataset size : {len(dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    iter_length = len(train_dataloader)
    model = create_model(
        'edgenext_xx_small_bn_hs',
        pretrained=True,
        num_classes=n_classes,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        input_res=input_size,
        classifier_dropout=0.0,
    )
    model.head = nn.Sequential(
        nn.Linear(168, 100),
        nn.BatchNorm1d(100),
        nn.Hardswish(),
        nn.Linear(100, n_classes)
    )

    model.to(device)
    print("Model configuration : ")
    print(
        pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size).to(device), show_input=True))
    # Criterion, Optimizer, Loss history tracker
    criterion = WeightedMSELoss(n_classes).to(device)
    #criterion1 = nn.CrossEntropyLoss().to(device)
    #criterion2 = nn.HuberLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0, last_epoch=- 1, verbose=True)
    # Create directory to save checkpoints
    os.makedirs(f"./checkpoint/edgenet_cls_{n_classes}/", exist_ok=True)

    print("Current CPU random seed :", torch.initial_seed())
    print("Current CUDA random seed :", torch.cuda.initial_seed())

    # Train model
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
            loss = criterion(prediction, cls_label, reg_label.view(-1, 1))
            #value = torch.argmax(prediction, dim=1)/n_classes
            #loss = 0.5*criterion1(prediction, cls_label) + 0.5*criterion2(value.view(-1, 1), reg_label.view(-1, 1))
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch #{epoch + 1} [{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss / (i + 1):.6f}, >>>> Training Accuracy : {100*(correct/total):.6f}")
        
        hist['train_loss'] += [training_loss/len(train_dataloader)]
        hist['train_acc'] += [100*(correct/total)]
        #visualize_cls(model, input_size, device, epoch, n_classes)
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
                loss = criterion(prediction, cls_label, reg_label.view(-1, 1))
                #value = torch.argmax(prediction, dim=1)/n_classes
                #loss = 0.5*criterion1(prediction, cls_label) + 0.5*criterion2(value.view(-1, 1), reg_label.view(-1, 1))
                val_loss += loss.item()
            
            print(
                    f"Epoch #{epoch + 1} [{i}/{len(val_dataloader)}] >>>> Validation loss : {val_loss / len(val_dataloader):.6f}, >>>> Validation Accuracy : {100*(correct/total):.6f}")
            hist['val_loss'] += [val_loss/len(val_dataloader)]
            hist['val_acc'] += [100*(correct/total)]
            torch.save(model, f"./checkpoint/edgenet_cls_{n_classes}/checkpoint_{epoch}.pt")

    plt.figure(figsize=(24, 7))
    plt.subplot(1, 2, 1)
    plt.plot(hist['train_loss'], 'k', linewidth=2, label="Training loss")
    plt.plot(hist['val_loss'], 'k--', linewidth=2, label="Validation loss")
    plt.legend(fontsize=15)

    plt.subplot(1, 2, 2)
    plt.plot(hist['train_acc'], 'k', linewidth=2, label="Training Accuracy")
    plt.plot(hist['val_acc'], 'k--', linewidth=2, label="Validation Accuracy")
    plt.legend(fontsize=15)
    plt.savefig(f"./result/graph_loss_edgenet_cls_{n_classes}.png")
    plt.close()
    
    model = torch.load(f"./checkpoint/edgenet_cls_{n_classes}/checkpoint_49.pt")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    visualize_cls(model, 112, device, cls_num=n_classes)