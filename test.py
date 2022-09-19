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



def test(cfg, args):
    '''Function for test w/ face blur detection model'''

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

    # (2) : training
    model_name = cfg['train']['model']

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

    # (1) : Device setting
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)

    # (2) : Create directory to save checkpoints
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.save + '/viz', exist_ok=True)

    # (3) : Resume previous training
    if '.ckpt' in args.resume or '.pt' in args.resume:
        print("RESUME")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    ##########################################################
    #                     START EVALUATION !!                #
    ##########################################################
    print(type(model))

    model.eval()
    epoch = 0
    with torch.no_grad():
        if task_name == 'regression':
            cos_mean_fix, cos_mean_random = visualize(model, img_size, device, epoch, args.save + '/viz')
        elif task_name == 'classification':
            cos_mean_fix, cos_mean_random = visualize_cls(model, img_size, device, epoch, num_classes, args.save + '/viz')

        fix_save_path = f'{args.save}/{exp_name}_cos_mean_fix.pkl'
        random_save_path = f'{args.save}/{exp_name}_cos_mean_random.pkl'

        with open(fix_save_path, 'wb') as f:
            pickle.dump(cos_mean_fix, f, pickle.HIGHEST_PROTOCOL)            
        f.close()

        with open(random_save_path, 'wb') as f:
            pickle.dump(cos_mean_random, f, pickle.HIGHEST_PROTOCOL)
        f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_regression', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--save', type=str, default='checkpoint/base_regression', help='Path to save model file')
    parser.add_argument('--resume', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open('config/' + args.config + '.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    test(cfg, args)

        
