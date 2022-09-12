# TODO random + fix
import os
from tqdm import tqdm
import cv2
import numpy as np
import random
import argparse

def blurring(img, d=None, angle=None):
    if d is None:
        random_degree = random.randint(0, 100)
    else:
        random_degree = d

    if angle is None:
        random_angle = random.randint(-180, 180)
    else:
        random_angle = angle

    if random_degree == 1:
        random_angle = random.randint(-88, 88)

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
        image = np.array(img)
        blurred = cv2.filter2D(image, -1, kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

    return blurred, random_degree/100

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_root', type=str, help="FFHQ1024 dataset dir")
    parser.add_argument('--mode', type=str, default='random', 'Select type of data samples. It can be "fix" or "random"')
    args = parser.parse_args()

    image_root =  args.img_root

    look_upto = 0
    for file in os.listdir(image_root):
        if os.path.splitext(file)[-1] not in ['.png', '.jpg']:
            continue

        look_upto += 1
        image_name = os.path.splitext(file)[0]
        os.makedirs(os.path.join(image_root, image_name), exist_ok=True)
        clean_img = cv2.imread(os.path.join(image_root, file))
        print(f"Creating blur images [{look_upto}/30]")
        for i in tqdm(range(101)):
            blurred_img, _ = blurring(clean_img, d=i, angle=None if args.mode == 'random' else 45)
            cv2.imwrite(os.path.join(image_root, image_name, f'{mode}_{i}.png'), blurred_img)

        if look_upto == 30:
            break
