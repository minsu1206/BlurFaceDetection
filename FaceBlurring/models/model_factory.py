import os
import sys
import argparse
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from models.mobilenet import MobileNet
from models.resnet import Resnet, ResnetCLS
from models.edgenext import EdgenextXXSmall
from models.efficientnet import EfficientNetLite

def model_build(model_name:str, num_classes:int):
    """
    <args>
        model_name : from config/@.yaml
        resume:
            if '' : no pretrained model, no resume
            else : use .ckpt file and resume the model
    """

    model_name = model_name.lower()

    ######      build model     ######
    # define class or def for building model at each [@ Net].py
    # just add if ~ : ~ code for another model like below.
    #

    if model_name == 'resnet':
        model = Resnet(num_classes=num_classes)
    
    if model_name == 'resnet_cls':
        model = ResnetCLS(num_classes=num_classes)

    if model_name == 'UEGAN':
        model = UEGAN() # UEGAN 파일 수정 필요

    if model_name == 'edgenext_xx_small':
        model = EdgenextXXSmall(num_classes=num_classes)

    if model_name == 'efficientnetlite':
        model = EfficientNetLite(num_classes=num_classes)

    if model_name == 'mobilenet':
        model = MobileNet(num_classes=num_classes)
        
        
    return model


def model_test(args):
    """
    (0) Build Model
    (1) Model size
    (2) Model inference time
    """

    # (0) Build Model
    model = model_build(args.model, args.cls)

    print(model)

    # (1) Model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    torch.save(model, f'./{args.model}.pth')
    print('pt file : ', round(os.path.getsize(f'./{args.model}.pth') / 1024**2, 3), 'MB')
    os.system(f'rm ./{args.model}.pth')

    # (2) Model inference time
    dummy_input = torch.zeros((1, 3, args.input_size, args.input_size))
    records = []
    rep = 200
    for i in range(200):
        start = time.process_time_ns()
        dummy_output = model(dummy_input)
        end = time.process_time_ns()
        records.append(end - start)
        time.sleep(0.0001)

    avg_speed_ns = sum(records) / rep
    std_speed = float(np.std(records))
    print("Inference AVG speed : ", round(avg_speed_ns / 1e6, 4), "(ms)")
    print("Inference STD speed : ", round(std_speed / 1e6, 4), "(ms)")
    print("Inference fastest speed : ", round(min(records) / 1e6, 4), "(ms)")
    print("Inference slowest sppeed : ", round(max(records) / 1e6, 4), "(ms)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cls', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=112)
    args = parser.parse_args()

    model_test(args)
