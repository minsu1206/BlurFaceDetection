import torch
try:
    from models.resnet import *
    from models.edgenext import *
    # from models.cspdarknet53 import *
    from models.yolov5 import *
    from models.squeezenet import *
    from models.mobilenetv2 import MobileNetV2
    from models.efficientnet import EfficientNetLite
    from models.mobilenetv3 import mobilenetv3
except:
    from resnet import *
    from edgenext import *
    # from cspdarknet53 import *
    from yolov5 import *
    from squeezenet import *
    from mobilenetv2 import MobileNetV2
    from efficientnet import EfficientNetLite
    from mobilenetv3 import mobilenetv3
import argparse
import time
import os
import numpy as np

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

    if model_name == 'resnet18':
        model = ResNet(block=ResidualBlock, num_block=[2, 2, 2, 2], num_classes=num_classes)
    
    if model_name == 'resnet34':
        model = ResNet(block=ResidualBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)
    
    if model_name == 'resnet50':
        model = ResNet(block=BottleNeckResidualBlock, num_block=[3, 4, 6, 3], num_classes=num_classes)
    
    if model_name == 'UEGAN':
        model = UEGAN()

    if model_name == 'edgenext_xx_small':
        model = edgenext_xx_small(num_classes=num_classes)
    
    # if model_name == 'cspdarknet53':
    #     model = CSPDarknet53(
    #         num_classes=num_classes, stem_channels=16,
    #         feature_channels=[64, 64, 64, 64, 64],
    #         branches=[2, 4, 4, 2]
    #     )
    
    if model_name == 'efficientnetlite':
        model = EfficientNetLite(num_classes=num_classes)

    if model_name == 'mobilenetv2_0.25':
        model = MobileNetV2(num_classes=num_classes, width_mult=0.25)
    
    if model_name == 'mobilenetv2_0.5':
        model = MobileNetV2(num_classes=num_classes, width_mult=0.5)

    if model_name == 'yolov5n':
        model = Model('yolov5n.yaml')
    
    if model_name == 'squeezenet1_1':
        model = squeezenet1_1(num_classes=num_classes)
        
    if model_name == 'mobilenetv3_0.5':
        model = mobilenetv3(num_classes=num_classes, width_mult=0.5)

    if model_name == 'mobilenetv3_0.75':
        model = mobilenetv3(num_classes=num_classes, width_mult=0.75)

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
    rep = 500
    for i in range(500):
        start = time.process_time_ns()
        dummy_output = model(dummy_input)
        end = time.process_time_ns()
        records.append(end - start)
        time.sleep(0.0001)

    avg_speed_ns = sum(records) / rep
    std_speed = float(np.std(records))
    print("Output shape : ", dummy_output.shape)
    print("Inference AVG speed : ", round(avg_speed_ns / 1e6, 4), "(ms)")
    print("Inference STD speed : ", round(std_speed / 1e6, 4), "(ms)")
    print("Inference fastest speed : ", round(min(records) / 1e6, 4), "(ms)")
    print("Inference slowest speed : ", round(max(records) / 1e6, 4), "(ms)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cls', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=112)
    args = parser.parse_args()

    model_test(args)


