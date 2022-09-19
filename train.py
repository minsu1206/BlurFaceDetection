import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import pytorch_model_summary
import torchvision.transforms as transforms

from dataset.dataset import FaceDataset, FaceDatasetVal
from utils import *
from model_factory import model_build
from loss import *



def train(cfg, args):
    '''Function for training face blur detection model'''

    ##########################################################
    #                     configuration                      #
    ##########################################################

    # (0) : global
    exp_name = cfg['exp_name']
    task_name = cfg['task_name']
    num_classes = None
    if 'num_classes' in cfg:
        num_classes = cfg['num_classes']
    device = args.device

    # (1) : dataset
    batch_size = cfg['dataset']['batch']
    img_size = cfg['dataset']['image_size']
    train_csv_path = cfg['dataset']['train_csv_path']
    dataset_metric = cfg['dataset']['metric']
    val_csv_path = cfg['dataset']['val_csv_path']

    # (2) : training
    model_name = cfg['train']['model']

    ##########################################################
    #                     Dataloader                         #
    ##########################################################
    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5)
                            ])

    total_dataset = FaceDataset(
        train_csv_path, dataset_metric, transform, img_size, 'rgb', task=task_name, num_classes=num_classes)
    
    if len(val_csv_path) == 0 :
        dataset_size = len(total_dataset)
        train_size = int(dataset_size*0.8)
        val_size = dataset_size - train_size
        train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    else :
        train_dataset = total_dataset
        val_dataset = FaceDatasetVal(
            val_csv_path, dataset_metric, transform, img_size, 'rgb', task=task_name,num_classes=num_classes)
    
    # Check number of each dataset size
    print(f"Training dataset size : {len(train_dataset)}")
    print(f"Validation dataset size : {len(val_dataset)}")
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ##########################################################
    #                     Build Model                        #
    ##########################################################

    if task_name == 'classification':
        model = model_build(model_name=model_name, num_classes=num_classes)
    else:
        model = model_build(model_name=model_name, num_classes=1)

    print("Model configuration : ")
    print(pytorch_model_summary.summary(model,
                                torch.zeros(batch_size, 3, img_size, img_size),
                                show_input=True))

    # only predict blur regression label -> num_classes = 1

    ##########################################################
    #                     Training SetUp                     #
    ##########################################################

    # (0) : Loss function
    loss_func = build_loss_func(cfg['train']['loss'], device=device)

    # (1) : Optimizer & Scheduler
    optimizer = build_optim(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    start = 0
    epochs = cfg['train']['epochs']

    # (2) : Device setting
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)

    # (3) : Create directory to save checkpoints
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.save + '/viz', exist_ok=True)

    # (4) : Resume previous training
    if '.ckpt' in args.resume or '.pt' in args.resume:
        print("RESUME")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optimizer.load_state_dict(checkpoint['optimizers_state_dict'])
        scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start = checkpoint['epoch']

    ##########################################################
    #                     START TRAINING !!                  #
    ##########################################################
    for epoch in range(start, epochs):

        # (0) : Training
        training_loss = 0.0
        model.train()

        for i, batch in tqdm(enumerate(train_dataloader)):
            
            optimizer.zero_grad()

            # Dataloader gives - regression:1 label // classification: 2 label
            gt_reg, gt_cls = None, None
            image = batch[0]
            image = image.to(device)
            
            if task_name == 'classification':   # classification
                gt_cls = batch[1][0]
                gt_cls = gt_cls.to(device)
                gt_reg = batch[1][1]
                gt_reg = gt_reg.to(device)

            elif task_name == 'regression':
                gt_reg = batch[1]
                gt_reg = gt_reg.to(device)

            prediction = model(image)

            loss = compute_loss(loss_func, prediction, gt_reg, gt_cls)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            # break

        print(f"Epoch #{epoch + 1} >>>> Training loss : {training_loss / len(train_dataloader):.6f}")

        scheduler.step()
        
        # (1): Evaluation
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, batch in tqdm(enumerate(val_dataloader)):

                gt_reg, gt_cls = None, None
                image = batch[0]
                image = image.to(device)
                
                if task_name == 'classification':
                    gt_cls = batch[1][0]
                    gt_cls = gt_cls.to(device)
                    gt_reg = batch[1][1]
                    gt_reg = gt_reg.to(device)
                elif task_name == 'regression':
                    gt_reg = batch[1]
                    gt_reg = gt_reg.to(device)
                
                prediction = model(image)

                loss = compute_loss(loss_func, prediction, gt_reg, gt_cls)
                validation_loss += loss.item()
                # break

            print(f"Epoch #{epoch + 1} >>>> Validation loss : {validation_loss / len(val_dataloader):.6f}")

        # (2) : Visualization
        if args.viz:
            if task_name == 'regression':
                cos_mean_fix, cos_mean_random = visualize(model, img_size, device, epoch, args.save + '/viz')
            elif task_name == 'classification':
                cos_mean_fix, cos_mean_random = visualize_cls(model, img_size, device, epoch, num_classes, args.save + '/viz')

            cos_mean_fix = round(float(np.mean(cos_mean_fix)), 4)
            cos_mean_random = round(float(np.mean(cos_mean_random)), 4)

            print(f"Epoch #{epoch + 1} >>>> Test Metric - fix: {cos_mean_fix}")
            print(f"Epoch #{epoch + 1} >>>> Test Metric - random: {cos_mean_random}")

        # (3) : Checkpoint
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch
            }
            , f"{args.save}/checkpoint_{epoch}.ckpt")
        print(f"Epoch #{epoch + 1} >>>> SAVE .ckpt file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_regression', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--save', type=str, default='checkpoint/base_regression', help='Path to save model file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    parser.add_argument('--viz', action='store_true')
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    train(cfg, args)
