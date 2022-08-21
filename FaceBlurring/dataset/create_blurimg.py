import argparse
from tqdm import tqdm
import pandas as pd
from blur import *
from utils import *
from insightface.app import FaceAnalysis
from facenet_pytorch import InceptionResnetV1
import torch

class CreateBlurImg:
	'''
		class to create blur image dataset from raw images
	'''
	def __init__(self, data_dir, blur_method='defocus', motionblur_hyperparameters=None, device=None):
		'''
			create blurred images in the data_directory
		'''
		self.device = 'cpu' if device is None else device
		# Available img files
		self.available = ['.png', '.jpg', 'PNG', 'JPG', 'JPEG']
		
		# motion blur method
		assert blur_method in ['defocus', 'deblurGAN']
		self.blur_method = blur_method

		# Get sample paths in list
		self.sample_paths = self._get_all_imgs(data_dir)
		self._create_sample_dirs()

		# padding option to face detection
		self.pad_max = 200

		if self.blur_method == 'defocus' or self.blur_method is None:
			# Get motion blur hyperparameters
			if motionblur_hyperparameters is None:
				self.parameters = {'mean':50, 'var':20, 'dmin':0, 'dmax':100}
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
				self.parameters = motionblur_hyperparameters
		else:
			raise ValueError(f'{blur_method} is not an available blur method')

	def _get_all_imgs(self, root):
		'''
			Function to get all image samples inside the directory
			os.walk will search all directory
		'''
		paths = []
		print('Check all sample images(clean)...')
		for (path, directory, files) in tqdm(os.walk(root)):
			for filename in files:
				# print(filename)
				ext = os.path.splitext(filename)[-1]
				if ext in self.available and 'clean' in path:
					paths += [os.path.join(path, filename)]
		return paths

	def _create_sample_dirs(self):
		print('Create sample directories...')
		for files in tqdm(self.sample_paths):
			# FIXME : [8/21] 경로 수정
			path = os.path.dirname(files)
			# print(path)
			# path2list = path.split(os.path.sep)
			# rootpath = os.path.sep.join(path2list[:3])
			# subpath = os.path.sep.join(path2list[4:])
			# blurpath = os.path.join(rootpath, 'blur_'+self.blur_method, subpath)
			# print(blurpath)
			blurpath = path.replace('clean', 'blur_' + self.blur_method)
			# print(blur)
			os.makedirs(blurpath, exist_ok=True)
			# break


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
		elif calc == 'cosine':
			metric = 'cosine'
			# FIXME : device
			resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
		else:
			raise ValueError("Not available metric.")
		
		dict_for_label = {'filename' : [], calc : []}
		if self.blur_method == 'defocus':
			for image_file in tqdm(self.sample_paths):
				if os.path.isfile(image_file):
					image = cv2.imread(image_file)
					if scrfd:
						pad=0
						find = False
						while not find and pad <= self.pad_max:
							padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
							face_image, find = crop_n_align(app, padded)
							pad+=50

						if find:
							image = face_image
				else:
					continue

				blurred, degree = blurring(image, self.parameters)
				if save and label:
					# path = os.path.dirname(image_file)
					# path2list = path.split(os.path.sep)
					# rootpath = os.path.sep.join(path2list[:3])
					# subpath = os.path.sep.join(path2list[4:])
					# blurpath = os.path.join(rootpath, 'blur_'+self.blur_method, subpath)

					# FIXME : 위 방식대로 하면 경로가 ../data 가 아닐때 안 먹어서 바꿈
					blurpath = image_file.replace('clean', 'blur_'+self.blur_method)
					# assert len(path)+len(self.blur_method) == len(blurpath), 'You should create data directory properly'
					cv2.imwrite(blurpath, blurred)
					
					# dict_for_label['filename'] += [os.path.join(blurpath, os.path.basename(image_file))]
					dict_for_label['filename'] += [blurpath]

					if callable(metric):
						dict_for_label[calc].append(metric(image, blurred))
					elif metric == 'degree':
						dict_for_label[calc].append(degree)
					elif metric == 'cosine':
						# [8/21] : CUDA / CPU both available
						clean_tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
						blur_tensor = torch.Tensor(blurred).permute(2, 0, 1).unsqueeze(0)
						emb_clean = resnet(clean_tensor if self.device is 'cpu' else clean_tensor.to(self.device))
						emb_blur = resnet(blur_tensor if self.device is 'cpu' else blur_tensor.to(self.device))
						emb_clean = emb_clean.squeeze(0).detach().numpy() if self.device is 'cpu' else emb_clean.cpu().squeeze(0).detach().numpy()
						emb_blur = emb_blur.squeeze(0).detach().numpy() if self.device is 'cpu' else emb_blur.cpu().squeeze(0).detach().numpy()
						cosine = np.dot(emb_clean, emb_blur)/(np.linalg.norm(emb_clean)*np.linalg.norm(emb_blur))
						dict_for_label[calc].append(1-cosine)

				elif save:
					path = os.path.dirname(image_file)
					path2list = path.split(os.path.sep)
					rootpath = os.path.sep.join(path2list[:3])
					subpath = os.path.sep.join(path2list[4:])
					blurpath = os.path.join(rootpath, 'blur_'+self.blur_method, subpath)

					assert len(path)+len(self.blur_method) == len(blurpath), 'You should create data directory properly'
					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)

				elif label:
					raise ValueError("You cannot save label without saving blur samples")

		elif self.blur_method == 'deblurGAN':
			for image_file in tqdm(self.sample_paths):
				self.parameters['expl'] = np.random.choice([0.003, 0.001,
										                    0.0007, 0.0005,
															0.0003, 0.0001])
				self.parameters['part'] = np.random.choice([1, 2, 3])
				trajectory = Trajectory(self.parameters).fit()
				psf, mag = PSF(self.parameters['canvas'], trajectory=trajectory).fit()
				image, blurred = BlurImage(image_file, psf, self.parameters['part'], scrfd, app).blur_image()
				if save and label:
					path = os.path.dirname(image_file)
					path2list = path.split(os.path.sep)
					rootpath = os.path.sep.join(path2list[:3])
					subpath = os.path.sep.join(path2list[4:])
					blurpath = os.path.join(rootpath, 'blur_'+self.blur_method, subpath)
					assert len(path)+len(self.blur_method) == len(blurpath), 'You should create data directory properly'
					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)
					
					dict_for_label['filename'] += [os.path.join(blurpath, os.path.basename(image_file))]
					if callable(metric):
						dict_for_label[calc].append(metric(image, blurred))
					elif metric == 'degree':
						dict_for_label[calc].append(mag)
					elif metric == 'cosine':
						emb_clean = resnet(torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)).squeeze(0).detach().numpy()
						emb_blur = resnet(torch.Tensor(blurred).permute(2, 0, 1).unsqueeze(0)).squeeze(0).detach().numpy()
						cosine = np.dot(emb_clean, emb_blur)/(np.linalg.norm(emb_clean)*np.linalg.norm(emb_blur))
						dict_for_label[calc].append(1-cosine)

				elif save:
					path = os.path.dirname(image_file)
					path2list = path.split(os.path.sep)
					rootpath = os.path.sep.join(path2list[:3])
					subpath = os.path.sep.join(path2list[4:])
					blurpath = os.path.join(rootpath, 'blur_'+self.blur_method, subpath)

					assert len(path)+len(self.blur_method) == len(blurpath), 'You should create data directory properly'
					cv2.imwrite(os.path.join(blurpath, os.path.basename(image_file)), blurred)
				
				elif label:
					raise ValueError("You cannot save label without saving blur samples")

		else:
			pass

		if label:
			print(dict_for_label)
			save_dir = ".."+os.path.sep+os.path.join('data', f"label_blur_{self.blur_method}", 'label')
			os.makedirs(save_dir, exist_ok=True)
			df = pd.DataFrame(dict_for_label)
			df.to_csv(os.path.join(save_dir, "label.csv"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This program creates blur images.')
	parser.add_argument('--blur', type=str, help='defocus, deblurGAN is available', default='defocus')
	parser.add_argument('--save', action='store_true', help='option to save blurred images')
	parser.add_argument('--label', action='store_true', help='option to create labels')
	parser.add_argument('--calc', type=str, help='option to make label(metrics), psnr, ssim, degree, cosine is available', default='psnr')
	parser.add_argument('--scrfd', action='store_true', help='Apply scrfd crop and align on the image')
	parser.add_argument('--root', type=str, help='Clean Image folder directory path', default='../data')
	parser.add_argument('--device', type=str, help='Use CUDA or Not', default='')	# CPU: '' // CUDA: 'cuda:0'
	args = parser.parse_args()

	blurrer = CreateBlurImg(args.root, args.blur, device=None if args.device =='' else args.device)
	blurrer.generate_blur_images(args.save, args.label, args.calc, args.scrfd)
