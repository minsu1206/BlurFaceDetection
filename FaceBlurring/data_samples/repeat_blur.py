import random
import os

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from scipy import signal
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def blurring(img, d, angle=None, n=1):
    '''
        Apply motion blur to the image(from defocus)
        img : source img
        param(dictionary) : [mean, var, dmin, dmax]
    '''
    H = img.shape[0]
    factor = 1024//H

    random_degree = int(d/factor)

    if random_degree == 1:
        random_angle = random.randint(-88, 88)
    else:
        random_angle = random.randint(-180, 180)

    if angle is not None:
        random_angle = angle

    if random_degree == 0:
        image = np.array(img)
        cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(image, dtype=np.uint8)

    else:
        # Create random motion blur kernel
        M = cv2.getRotationMatrix2D((random_degree / 2, random_degree / 2), random_angle, 1)
        kernel = np.diag(np.ones(random_degree))
        kernel = cv2.warpAffine(kernel, M, (random_degree, random_degree))
        kernel = kernel / random_degree

        # Apply kernel on the image sample
        blurred = np.array(img)
        for _ in range(n):
            blurred = cv2.filter2D(blurred, -1, kernel)
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            blurred = np.array(blurred, dtype=np.uint8)

    return blurred


if __name__ == '__main__':
    img_src = cv2.cvtColor(cv2.imread('/data/faceblur/BlurFaceDetection/FaceBlurring/data/FFHQ_1024/clean/00000/00014.png'), cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_src, (256, 256), interpolation=cv2.INTER_AREA)
    blurred = blurring(img_src, 300, 0)
    plt.imshow(blurred)
    plt.show()