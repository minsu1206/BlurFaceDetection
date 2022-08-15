import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from utils import *

def compare_and_plotimgs(image1, image2, metric, name=''):
	assert callable(metric), 'This metric is not supported.'
	value = metric(image1, image2)

	compare_img = np.concatenate((image1, image2), 1)
	plt.figure(figsize=(10, 5))
	plt.imshow(compare_img)
	plt.title(f"{metric.__name__.upper()} value : {value : .2f}", fontsize=20)
	plt.axis('off')
	plt.savefig(f'../results/{metric.__name__.upper()}/sample{name}.png')
	plt.close()
	return

if __name__ == "__main__":
	metric = ssim
	
	os.makedirs(f'../results/{metric.__name__.upper()}', exist_ok=True)

	iter_root1 = '../data/sample_root/clean/00000/'
	iter_root2 = '../data/sample_root/blur/00000/'
	image1 = []
	image2 = []

	for img_pth in tqdm(os.listdir(iter_root1)):
		image1 += [cv2.cvtColor(cv2.imread(os.path.join(iter_root1, img_pth)), cv2.COLOR_BGR2RGB)]
	for img_pth in tqdm(os.listdir(iter_root2)):
		image2 += [cv2.cvtColor(cv2.imread(os.path.join(iter_root2, img_pth)), cv2.COLOR_BGR2RGB)]

	for i in tqdm(range(len(iter_root1))):
		img1, img2 = image1[i], image2[i]
		compare_and_plotimgs(img1, img2, metric, '_'+str(i))

