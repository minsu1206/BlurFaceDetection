import cv2
import matplotlib.pyplot as plt
from blur import *

if __name__ == '__main__':
    sample = '/data/faceblur/BlurFaceDetection/FaceBlurring/data_samples/data/FFHQ_1024/clean/00008/fix_0.png'
    og = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)
    blur1, kernel = blurring(og, {'dmin':0, 'dmax':100, 'mean':50, 'var':20})

    parameters = {'canvas':64, 'iters':2000, 'max_len':60,
                  'expl':np.random.choice([0.003, 0.001,
                                           0.0007, 0.0005,
                                           0.0003, 0.0001]),
                  'part':np.random.choice([1, 2, 3])}

    trajectory = Trajectory(parameters).fit()
    psf, mag = PSF(parameters['canvas'], trajectory=trajectory).fit()
    image, blurred = BlurImage(sample, psf, 3).blur_image()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(og)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(blur1)
    plt.axis('off')
    plt.savefig('linear.png')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(og)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(psf[3], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig('nonlinear.png')