import cv2
import numpy as np
import os
from tqdm import tqdm
import imutils
import matplotlib.pyplot as plt

# OpenCV method(FFT)
def detect_blur_fft(image, size=60):
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    return mean

# OpenCV method(Laplacian)
def calculate_focal_measure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

if __name__ == '__main__':
    path = './data/FFHQ_1024/clean/'
    look_upto = 0
    fft_mean, focal_mean = np.zeros(100), np.zeros(100)

    for subpath in os.listdir(path):
        if os.path.splitext(subpath)[-1] not in ['.png', '.jpg']:
            sample_path = os.path.join(path, subpath)
            look_upto += 1
            for i in tqdm(range(1, 101)):
                blur = cv2.imread(os.path.join(sample_path, f'fix_{i}.png'))
                blur = imutils.resize(blur, width=500)
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
                m = detect_blur_fft(gray, size=60)
                fm = calculate_focal_measure(blur)
                fft_mean[i-1] += m
                focal_mean[i-1] += fm
            if look_upto == 30:
                break

    fft_mean /= look_upto
    focal_mean /= look_upto

    plt.figure(figsize=(12, 7))
    plt.plot(fft_mean, 'k-', linewidth=2)
    plt.title("Blur detection(FFT method)", fontsize=20)
    plt.savefig("fix_fft.png")

    plt.figure(figsize=(12, 7))
    plt.plot(focal_mean, 'k-', linewidth=2)
    plt.title("Blur detection(Laplacian method)", fontsize=20)
    plt.savefig("fix_laplacian.png")