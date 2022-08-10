import argparse
import pandas as pd
import numpy as np
import math, cv2, os
from torch.utils.data import Dataset
from utils import *
import matplotlib.pyplot as plt

class FaceDataset(Dataset):
	def __init__(self, txt_file, option='clean', calc='psnr'):
		'''
			face dataset module
			txt file must include root directory of sample images
		'''
		self.calc = calc
		assert calc in ['psnr', 'ssim', 'degree'], "Not available metric"

		with open(txt_file, 'r') as f:
			lines = f.readlines()
			sample_root = [l.rstrip('\n') for l in lines]
	
		if option=='clean':
			self.sample_paths = self._get_clean_samples(sample_root)
			self.labels = np.zeros(len(self.sample_paths))

		elif option == 'blur':
			self.sample_paths, self.labels = self._get_blur_samples(sample_root)
		else:
			raise ValueError("option should be 'clean' or 'blur'")

	def _get_clean_samples(self, roots):
		'''
			Inner function to get all clean samples under sample root
			This function only return clean images
		'''
		paths = []
		for root in roots:
			for (path, directory, files) in os.walk(root):
				for filename in files:
					ext = os.path.splitext(filename)[-1]
					if ext in ['.png', '.jpg', 'PNG', 'JPG', 'JPEG'] and 'clean' in path:
						paths += [os.path.join(path, filename)]
		return paths

	def _get_blur_samples(self, roots):
		'''
			Inner function to get all blur samples under sample root
			This function only return blur images
		'''
		paths = []
		labels = []
		label_path = "../data/label/label.csv"
		assert os.path.isfile(label_path), "label file does not exist"
		df = pd.read_csv(label_path)
		assert self.calc in list(df.columns.values), 'Regenerate label with same metric'

		for root in roots:
			for (path, directory, files) in os.walk(root):
				for filename in files:
					ext = os.path.splitext(filename)[-1]
					if ext in ['.png', '.jpg', 'PNG', 'JPG', 'JPEG'] and 'blur' in path:
						filepath = os.path.join(path, filename)
						paths += [filepath]
						labels.append(df.loc[df['filename'] == filepath][self.calc].item())
						
		return paths, labels


	def __len__(self):
		return len(self.sample_paths)

	def __getitem__(self, idx):
		img_path, label = self.sample_paths[idx], self.labels[idx]
		image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
		return image, label

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This program makes dataset for face samples')
	parser.add_argument('--source', type=str, help='Needs txt file containing roots', default = '../config/test.txt')
	parser.add_argument('--option', type=str, help='choose between clean and blur', default='blur')
	parser.add_argument('--calc', type=str, help='choose between psnr, ssim, degree', default='psnr')
	args = parser.parse_args()
	dataset = FaceDataset(args.source, args.option, args.calc)

	plt.figure(figsize=(20, 20))
	for i in range(len(dataset)):
		image, label = dataset[i]
		plt.subplot(5, 5, i+1)
		plt.imshow(image)
		plt.axis('off')
		plt.title(f'Label : {label:.4f}', fontsize=16)
		if i==24:
			break
	name=input("Enter anything to save the img(if do not want to save, enter 'no') :")
	if name != 'no':
		plt.savefig(f'result_{name}.png')
	plt.show()
