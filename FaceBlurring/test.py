import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse

import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from models.mobilenet import FaceMobileNet
# import pytorch_model_summary
from insightface.app import FaceAnalysis

from dataset.dataset import FaceDataset
from models.model_factory import model_build
import pytorch_lightning as pl
from dataset.blur import crop_n_align


# def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss):
#     progress = int(iteration/(num_data // batch_size)*100//4)
#     print(
#         f"Epoch : {epoch}/{num_epoch} >>>> train : {iteration}/{num_data // batch_size}{iteration / (num_data//batch_size) * 100:.2f}"
#         + '=' * progress + '>' + ' ' * (25 - progress) + f") loss : {loss: .6f}", end='\r')



def test(cfg, args, mode):


    ##############################
    #       BUILD MODEL          #
    ##############################


    model = model_build(model_name=cfg['train']['model'], num_classes=1)
    # only predict blur regression label -> num_classes = 1
    
    if '.ckpt' or '.pt' in args.resume:
        model_state = torch.load(args.resume)
        model = model.load_state_dict(model_state)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)
    

    ##############################
    #       MODE : VIDEO         #
    ##############################


    if mode == 'video':
        save_path = args.save_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path, fourcc, 25.0, (640,480))

        video_path = args.file_path
        cap = cv2.VideoCapture(video_path)
        width  = cap.get(3) # width
        height = cap.get(4) # height

        app = FaceAnalysis(allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        while(cap.isOpened()):
            ret, frame = cap.read()

            pad = 0
            find = False


            while not find and pad <= 200:
                padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
                face_image, find, faces = crop_n_align(app, padded, box=True)
                pad += 50

                if find:
                    # blur_label = f'{model(face_image):.2f}'
                    blur_label = f'{np.random.rand(1)[0]:.2f}'
                else:
                    blur_label = 'Face not found'

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            if blur_label == 'Face not found':
                TextPosition = (int(width*0.4), int(height*0.9))
            else:
                TextPosition = (int(width*0.45), int(height*0.9))
            fontScale              = 1
            fontColor              = (255,255,255)
            thickness              = 2
            lineType               = 2

            cv2.putText(frame,blur_label, 
                TextPosition, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            if len(faces) != 0:
                bbox = faces[0]['bbox']
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()



    ##############################
    #       MODE : IMAGE         #
    ##############################


    if mode == 'image':
        raise NotImplementedError()
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--file_path', type=str, default='./data/sample.mp4')
    parser.add_argument('--save_path', type=str, default='./data/blur_sample.avi')
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mode = args.mode.lower()
    assert mode in ['video', 'image']

    test(cfg, args, mode)

    


    
