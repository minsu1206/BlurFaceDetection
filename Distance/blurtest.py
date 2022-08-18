import cv2
import numpy as np
import random

# Blurring from defocus(manually control degree and angle)
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