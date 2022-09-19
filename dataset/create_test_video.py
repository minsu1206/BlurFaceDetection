import argparse

import cv2
import numpy as np
from pytube import YouTube
from pytube.cli import on_progress
from insightface.app import FaceAnalysis

from utils import crop_n_align


def download_youtube_video(url, save_path):
    print("Download youtube video")
    yt = YouTube(url, on_progress_callback=on_progress)
    stream = yt.streams.get_highest_resolution()
    stream.download(filename=save_path)


def cut_video(file_path, save_path):
    print("Extract only the face from the video")
    cap = cv2.VideoCapture(file_path, apiPreference=cv2.CAP_MSMF)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    idx = 0

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
            face_image, find = crop_n_align(app, padded)
            if find:
                break
            pad += 50

        if find:
            out.write(frame)
            idx = 0
        elif idx <= 60:
            out.write(frame)
            idx += 1
        else:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, default='https://www.youtube.com/watch?v=yp4GmpmFyPY&ab_channel=BenParkes', help='Youtube url for the video you want to download')
    parser.add_argument('--file_path', type=str, default='../data/sample.mp4', help='Path for the origin video')
    parser.add_argument('--save_path', type=str, default='../data/edited_sample.mp4', help='Path for the edited video')
    args = parser.parse_args()

    download_youtube_video(args.url, args.file_path)
    cut_video(args.file_path, args.save_path)