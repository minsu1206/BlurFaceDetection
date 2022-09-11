import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_model_summary
import torchvision.transforms as transforms

from dataset.dataset2 import FaceDataset, FaceDatasetVal
from utils import *
from models.model_factory import model_build

# TODO : debugging
def build_loss_func(loss_dict, device):
    loss_compute_dict = {}
    for key, val in loss_dict.items():
        key = key.lower()
        if key == 'huber':
            func = nn.HuberLoss()
        
        if key == 'crossentropy':
            func = nn.CrossEntropyLoss()
        
        if key == 'mse':
            func = nn.MSELoss()

        if key == 'l1':
            func = nn.L1Loss()

        if key == 'prob+mse':
            func = prob_based_mse()

        weight = val

        loss_compute_dict[key] = {'func': func.to(device), 'weight': weight}

    return loss_compute_dict

# TODO : debugging
def compute_loss(loss_func, pred, gt_reg, gt_cls):
    total_loss = 0
    for loss_name, loss_dict in loss_func.items():
        loss = loss_dict['func'](pred, gt_cls)
        loss *= loss_dict['weight']
        total_loss += loss
    
    return total_loss

# TODO : prob_based mse error



def train(cfg, args):
    '''Function for training face blur detection model'''

    ##############################
    #       configuration        #
    ##############################
    # (0) : global
    exp_name = cfg['exp_name']
    task_name = cfg['task_name']
    num_classes = None
    if 'num_classes' in cfg['task_name']:
        num_classes = cfg['task_name']['num_classes']
    
    # (1) : dataset
    batch_size = cfg['dataset']['batch']
    img_size = cfg['dataset']['image_size']
    train_csv_path = cfg['dataset']['train_csv_path']
    dataset_metric = cfg['dataset']['metric']
    val_csv_path = cfg['dataset']['val_csv_path']

    # (2) : training
    model_name = cfg['train']['model']

    ##############################
    #       DataLoader           #
    ##############################
    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5)
                            ])

    train_dataset = FaceDataset(train_csv_path, dataset_metric, transform, img_size, 'rgb', num_classes=num_classes)
    val_dataset = FaceDatasetVal(val_csv_path, dataset_metric, transform, img_size, 'rgb', num_classes=num_classes)
    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ##############################
    #       BUILD MODEL          #
    ##############################

    if task_name == 'classification':
        num_classes = cfg['task_name']['n_cls']
        model = model_build(model_name=model_name, num_classes=num_classes)
    else:
        model = model_build(model_name=model_name, num_classes=1)

    print("Model configuration : ")
    print(pytorch_model_summary.summary(model,
                                torch.zeros(batch_size, 3, img_size, img_size).to(device),
                                show_input=True))
    # only predict blur regression label -> num_classes = 1

    ##############################
    #       Training SetUp       #
    ##############################

    # loss / optim / scheduler / ...
    # if model_name == 'resnet_cls':
    #     loss_func = build_loss_func(cfg['train']['loss'])
    #     loss_func1, loss_func2 = loss_func[0], loss_func[1]
    # else:
    #     loss_func = build_loss_func()

    loss_func = build_loss_func(cfg['train']['loss'])

    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    epochs = cfg['train']['epochs']

    device = args.device
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)
        loss_func = loss_func.to(device)
    
    # Create directory to save checkpoints
    os.makedirs(args.save, exist_ok=True) # "./checkpoint/effnet_112_random/"

    # Continue previous training
    if '.ckpt' in args.resume:
        checkpoint = torch.load(args.resume)
        model = model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optimizer.load_state_dict(checkpoint['optimizers_state_dict'])
        scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    ##############################
    #       START TRAINING !!    #
    ##############################
    for epoch in range(epochs):
        training_loss = 0.0
        model.train()

        for i, batch in tqdm(enumerate(train_dataloader)):
            
            optimizer.zero_grad()

            # FIXME : regression = 1 label // classification = 2 label
            gt_reg, gt_cls = None, None
            image = batch[0]
            image = image.to(device)
            
            if task_name == 'classification':   # classification
                gt_cls = batch[1]
                gt_cls = gt_cls.to(device)
                gt_reg = batch[2]
                gt_reg = gt_reg.to(device)
            elif task_name == 'regression':
                gt_reg = batch[1]
                gt_reg = gt_reg.to(device)

            prediction = model(image)

            # TODO : compute loss integration
            # if model_name == 'resnet_cls':
            #     cls_label, reg_label = label
            #     value = torch.argmax(prediction, dim=1)*0.001
            #     loss = 0.5*loss_func1(prediction, cls_label) + \
            #             0.5*loss_func2(value, reg_label.view(-1, 1))
            # else:
            #     loss = loss_func(prediction, label.view(-1, 1))

            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch #{epoch + 1} >>>> Training loss : {training_loss / len(train_dataloader):.6f}")
        visualize(model, img_size, device, epoch)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, (image, label) in tqdm(enumerate(val_dataloader)):
                image, label = image.to(device), label.to(device)
                prediction = model(image)
                
                # TODO
                # if model_name == 'resnet_cls':
                #     cls_label, reg_label = label
                #     value = torch.argmax(prediction, dim=1)*0.001
                #     loss = 0.5*loss_func1(prediction, cls_label) + \
                #             0.5*loss_func2(value, reg_label.view(-1, 1))
                # else:
                #     loss = loss_func(prediction, label.view(-1, 1))
                validation_loss += loss.item()
            print(f"(Val)Epoch #{epoch + 1} >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}")


        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"{args.save}/checkpoint_{epoch}.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.yaml', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--save', type=str, default='./checkpoint/effnet_112_random/', help='Path to save model file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)
