import argparse

import yaml
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

from models.model_factory import model_build
from dataset.utils import crop_n_align

##########################################################
# WARNING !!!
# Not completed... It will be updated [9/12] 
##########################################################

def demo(cfg, args, mode):
    '''
    Test code for face blur detection
    
    Args:
        cfg: configuration file of yaml format
        args
        mode: inference mode. it can be "video" or "image"
    '''
    ##############################
    #       BUILD MODEL          #
    ##############################
    model = model_build(model_name=cfg['train']['model'], num_classes=1)
    # only predict blur regression label -> num_classes = 1
    
    if '.ckpt' or '.pt' in args.pretrained_path:
        model_state = torch.load(args.pretrained_path)
        model = model.load_state_dict(model_state)

    device = args.device
    if 'cuda' in device and torch.cuda.is_available():
        model = model.to(device)
    
    ##############################
    #       MODE : VIDEO         #
    ##############################
    if mode == 'video':
        video_path = args.file_path
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_MSMF)
        width  = int(cap.get(3))
        height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        save_path = args.save_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # for face detection
        app = FaceAnalysis(allowed_modules=['detection'],
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        while(cap.isOpened()):
            grabbed, frame = cap.read()
            if not grabbed:
                break

            pad = 0
            find = False
            
            while pad <= 200:
                padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
                face_image, find, faces = crop_n_align(app, padded, box=True)
                if find:
                    break
                pad += 50

            if find:
                blur_label = f'{model(face_image):.2f}' # predict blur label
            else:
                blur_label = 'Face not found'

            if len(faces) != 0:
                bbox = faces[0]['bbox']
                left_top = (int(bbox[0]-pad//2), int(bbox[1]-pad//2))
                right_btm = (int(bbox[2]-pad//2), int(bbox[3]-pad//2))
                red_color = (0, 0, 255)
                thickness = 3
                cv2.rectangle(frame, left_top, right_btm, red_color, thickness)

            if blur_label == 'Face not found':
                TextPosition = (int(width*0.38), int(height*0.9))
            else:
                TextPosition = (int(width*0.48), int(height*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255,255,255)
            thickness = 2
            lineType = 2

            cv2.putText(frame, blur_label, 
                TextPosition, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
            
            cv2.imshow('blur image', frame)
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
        image_path = args.file_path
        frame = cv2.imread(image_path)
        width, height = frame.shape[0], frame.shape[1]
        app = FaceAnalysis(allowed_modules=['detection'],
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        pad = 0
        find = False

        while pad <= 200:
            padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
            face_image, find, faces = crop_n_align(app, padded, box=True)
            if find:
                break
            pad += 50

            if find:
                blur_label = f'{model(face_image):.2f}'
            else:
                blur_label = 'Face not found'

        if len(faces) != 0:
            bbox = faces[0]['bbox']
            left_top = (int(bbox[0]-pad//2), int(bbox[1]-pad//2))
            right_btm = (int(bbox[2]-pad//2), int(bbox[3]-pad//2))
            red_color = (0, 0, 255)
            thickness = 3
            cv2.rectangle(frame, left_top, right_btm, red_color, thickness)

        if blur_label == 'Face not found':
            TextPosition = (int(width*0.4), int(height*0.9))
        else:
            TextPosition = (int(width*0.45), int(height*0.9))
        font = cv2.FONT_HERSHEY_SIMPLEX
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
        cv2.imshow('blur image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.yaml', help='Path for configuration file')
    parser.add_argument('--device', type=str, default='cpu', help='Device for model inference. It can be "cpu" or "cuda" ')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path for pretrained model file')
    parser.add_argument('--mode', type=str, default='video', help='Inference mode. it can be "video" or "image"')
    parser.add_argument('--file_path', type=str, default='./data/me4.mp4', help='Path for the video or image you want to infer')
    parser.add_argument('--save_path', type=str, default='./data/blur_sample.mp4', help='Path for saved the inference video')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mode = args.mode.lower()
    assert mode in ['video', 'image']

    demo(cfg, args, mode)
