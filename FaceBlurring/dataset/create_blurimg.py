import argparse
import json
from tqdm import tqdm
import pandas as pd
from blur import *
from utils import *
from insightface.app import FaceAnalysis

class CreateBlurImg:
	'''
		class to create blur image dataset from raw images
	'''
	def __init__(self, data_dir, blur_method='defocus', motionblur_hyperparameters=None):
		'''
			create blurred images in the data_directory
		'''
		# Available img files
		self.available = ['.png', '.jpg', 'PNG', 'JPG', 'JPEG']

		# Get sample paths in list
		self.sample_paths = self._get_all_imgs(data_dir)

		# motion blur method
		self.blur_method = blur_method
		
		if self.blur_method == 'defocus' or self.blur_method is None:
			# Get motion blur hyperparameters
			if motionblur_hyperparameters is None:
				self.parameters = {'degree' : 50}
			else:
				self.parameters = motionblur_hyperparameters

		elif self.blur_method == 'deblurGAN':
			if motionblur_hyperparameters is None:
				self.parameters = {'canvas':64,
					'iters':2000,
					'max_len':60,
					'expl':np.random.choice([0.003, 0.001,
							0.0007, 0.0005,
							0.0003, 0.0001]),
					'part':np.random.choice([1, 2, 3])}
			else:
				self.parameters = {'canvas':64,
					'iters':2000,
					'max_len':60,
					'expl':0.001,
					'part':2}
				self.parameters.update(motionblur_hyperparameters)
		else:
			raise ValueError(f'{blur_method} is not an available blur method')

		self._create_sample_dirs()

	def _get_all_imgs(self, root):
		'''
			Function to get all image samples inside the directory
			os.walk will search all directory
		'''
		paths = []
		print('Check all sample images(clean)...')
		for (path, directory, files) in tqdm(os.walk(root)):
			for filename in files:
				ext = os.path.splitext(filename)[-1]
				if ext in self.available and 'clean' in path:
					paths += [os.path.join(path, filename)]

		return paths

	def _create_sample_dirs(self):
		print('Create sample directories...')
		for files in tqdm(self.sample_paths):
			path = os.path.dirname(files)
			path2list = path.split('/')
			rootpath = '/'.join(path2list[:3])
			subpath = '/'.join(path2list[4:])
			if self.blur_method == 'defocus':
				blurpath = os.path.join(rootpath, f'blur_defocus_{self.parameters["degree"]}', subpath)
			elif self.blur_method == 'deblurGAN':
				blurpath = os.path.join(rootpath, f'blur_deblurGAN_{self.parameters["expl"]}_{self.parameters["part"]}', subpath)
			os.makedirs(blurpath, exist_ok=True)

	def generate_blur_images(self, save=True, label=False, calc='psnr', scrfd=False):
		# scrfd normalize and align
		app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
		app.prepare(ctx_id=0, det_size=(640, 640))
		print('Generate blur images...')
		if calc == 'psnr':
			metric = psnr
		elif calc == 'ssim':
			metric = ssim
		elif calc == 'degree':
			metric = 'degree'
		else:
			raise ValueError("Not available metric.")
		
		dict_for_label = {'filename' : [], calc : []}
		if self.blur_method == 'defocus':
			for image_file in tqdm(self.sample_paths):
				if os.path.isfile(image_file):
					image = cv2.imread(image_file)
					if scrfd:
						image, find = crop_n_align(app, image)
						if not find:
							continue
				else:
					continue
				blurred, degree = blurring(image, self.parameters)

				if save and label:
					path = os.path.dirname(image_file)
					path2list = path.split('/')
					rootpath = '/'.join(path2list[:3])
					subpath = '/'.join(path2list[4:])
					blurpath = os.path.join(rootpath, f'blur_defocus_{self.parameters["degree"]}', subpath)

					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)
					
					dict_for_label['filename'] += [os.path.join(blurpath, os.path.basename(image_file))]

					if callable(metric):
						dict_for_label[calc].append(metric(image, blurred))
					else:
						dict_for_label[calc].append(degree)

				elif save:
					path = os.path.dirname(image_file)
					path2list = path.split('/')
					rootpath = '/'.join(path2list[:3])
					subpath = '/'.join(path2list[4:])
					blurpath = os.path.join(rootpath, f'blur_defocus_{self.parameters["degree"]}', subpath)

					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)

				elif label:
					raise ValueError("You cannot save label without saving blur samples")

		elif self.blur_method == 'deblurGAN':
			for image_file in tqdm(self.sample_paths):
				trajectory = Trajectory(self.parameters).fit()
				psf, mag = PSF(self.parameters['canvas'], trajectory=trajectory).fit()
				image, blurred = BlurImage(image_file, psf, self.parameters['part'], scrfd, app).blur_image()
				if save and label:
					path = os.path.dirname(image_file)
					path2list = path.split('/')
					rootpath = '/'.join(path2list[:3])
					subpath ='/'.join(path2list[4:])
					blurpath = os.path.join(rootpath, f'blur_deblurGAN_{self.parameters["expl"]}_{self.parameters["part"]}', subpath)

					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)
					
					dict_for_label['filename'] += [os.path.join(blurpath, os.path.basename(image_file))]
					if callable(metric):
						dict_for_label[calc].append(metric(image, blurred))
					else:
						dict_for_label[calc].append(mag)

				elif save:
					path = os.path.dirname(image_file)
					path2list = path.split('/')
					rootpath = '/'.join(path2list[:3])
					subpath ='/'.join(path2list[4:])
					blurpath = os.path.join(rootpath, f'blur_deblurGAN_{self.parameters["expl"]}_{self.parameters["part"]}', subpath)

					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)
				
				elif label:
					raise ValueError("You cannot save label without saving blur samples")

		else:
			pass

		if label:
			save_dir = "../data/label/"
			os.makedirs(save_dir, exist_ok=True)
			df = pd.DataFrame(dict_for_label)
			df.to_csv(os.path.join(save_dir, "label.csv"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This program creates blur images.')
	parser.add_argument('--blur', type=str, help='defocus, deblurGAN is available', default='defocus')
	parser.add_argument('--save', type=bool, help='option to save blurred images', default='True')
	parser.add_argument('--label', type=bool, help='option to create labels', default='True')
	parser.add_argument('--calc', type=str, help='option to make label(metrics), psnr, ssim, degree is available', default='psnr')
	parser.add_argument('--scrfd', type=bool, help='Apply scrfd crop and align on the image', default=False)
	parser.add_argument('--hyperparam', type=json.loads, help='if blur type defocus : \'{"degree": 50}\', if blur type deblurGAN : \'{"expl": 0.001, "part":2}\'', default='{"degree": 50}')
	args = parser.parse_args()

	blurrer = CreateBlurImg("../data", args.blur, args.hyperparam)
	blurrer.generate_blur_images(args.save, args.label, args.calc, args.scrfd)
