import os
import argparse
import datetime

import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision

from dataset.dataset import FaceDataset
from models.model_factory import model_build
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

# TODO : Pytorch Lightning Wrapping -> Multi GPU


class FaceBlur(pl.LightningModule):
	'''
	A model that predicts the degree of blurring of an image

	Attributes:
		cfg: configuration file
		model: model that predicts the degree of blurring
		loss_group: dictionary for containing loss function
		resume: checkpoint where training info is stored

	Methods:
		_build_loss: add loss function to loss_group
		forward: receive the image and return the degree of blur
		compute_loss: calculate loss
		training_step: receive training batch data and return loss
		validation_step: receive validation batch data and return loss
		configure_optimizers: build optimizer and learning rate scheduler
	'''
	def __init__(self, cfg, args, resume):
		'''initialize loss function and model'''
		super().__init__()
		self.cfg = cfg
		self.model = model_build(model_name=cfg['train']['model'], num_classes=1)

		self.loss_group = {}
		self._build_loss()
		self.resume = resume
		if self.resume != None:
			self.model.load_state_dict(self.resume['model'])	# FIXME : debugging NOT YET

	def _build_loss(self):
		'''add loss function to loss_group'''
		keys = list(self.cfg['train']['loss'].keys())

		if 'MSE' in keys:
			self.MSE = nn.MSELoss()
			self.loss_group['MSE'] = self.MSE
		
		# TODO : add loss functions

	def forward(self, x):
		'''
		receive the image and return the degree of blur

		Args:
			x: images

		Returns:
			y: predicted blur labels
		'''
		y = self.model(x)
		# y = torch.sigmoid(y)
		return y
	
	def compute_loss(self, x, y):
		'''
		calculate the loss by receiving the predicted and target blur values

		Args:
			x: predicted blur labels
			y: target blur labels

		Returns:
			loss_dict: dictionary which contains loss values
		'''
		loss_dict = {"loss":0}

		for key, func in self.loss_group.items():
			loss_unit = func(x, y).float()
			loss_dict["loss"] += loss_unit
			loss_dict[key] = loss_unit.detach()
		
		return loss_dict

	def training_step(self, batch):
		'''
		receive training training batch data and return loss

		Args:
			batch: batch which contains images and target blur values

		Returns:
			loss_dict: dictionary which contains loss values for input batch
		'''
		x, y = batch
		x = x.float()       # image
		y = y.float()       # blur label
	
		pred = self.forward(x)

		loss_dict = self.compute_loss(pred, y)

		for key, val in loss_dict.items():
			self.log(f'train loss :: {key} : ', round(float(val.item()), 6),
			on_epoch=True, sync_dist=True, rank_zero_only=True)

		# NO USE print(~) --> too much verbose
		return loss_dict

	## if use more than 1 loss functions
	# def training_epoch_end(self, outputs):
	# 	train_epoch_loss = {}
	# 	for output in outputs:
	# 		for key, val in output.items():
	# 			if key in train_epoch_loss:
	# 				train_epoch_loss[key] += val.item()
	# 			else:
	# 				train_epoch_loss[key] = val.item()
		
	# 	for key, val in train_epoch_loss.items():
	# 		epoch_avg = val / len(outputs)
	# 		print(f'Train Epoch Loss : {key} : ', round(float(epoch_avg), 6))

	def validation_step(self, batch):
		'''
		receive validation validation batch data and return loss

		Args:
			batch: batch which contains images and target blur values

		Returns:
			loss_dict: dictionary which contains loss values for input batch
		'''
		x, y = batch
		x = x.float()
		y = y.float()

		pred = self.forward(x)

		loss_dict = self.compute_loss(pred, y)
		self.log('val_loss', loss_dict['MSE'])
		return loss_dict

	
	## if use more than 1 loss functions or visualize at validation
	# def validation_epoch_end(self, outputs):
	# 	raise NotImplementedError()

	def configure_optimizers(self):
		'''
		build optimizer and learning rate scheduler

		Returns: optimizer and learning rate scheduler		
		'''
		optim = build_optim(cfg, self.model)
		scheduler = build_scheduler(cfg, optim)

		if self.resume != None:
			optim.load_state_dict(self.resume['optimizer_states'])
			scheduler.load_state_dict(self.resume['lr_schedulers'])
			print('RESUME : optimizer / lr_scheduler')

		return [
			{
				"optimizer": optim,
				"lr_scheduler": {
					"scheduler": scheduler,
					"monitor": "metric_to_track"
				}
			}
		]

def train(cfg, args):
	'''Function for training face blur detection model'''
	now = datetime.datetime.now().strftime("%m_%d_%H")
	checkpoint_path = os.path.join(args.save, args.config, now) # path to save training information
	log_dir = os.path.join(checkpoint_path, 'log') # path to save training log
	os.makedirs(log_dir, exist_ok=True)
	logger = TensorBoardLogger(log_dir)
	
	##############################
	#       DataLoader           #
	##############################
	dataset = FaceDataset(
		label_path=cfg['dataset']['csv_path'],	data_root=cfg['dataset']['img_root'],
		transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(224)]), input_size=512
	)

	dataset_size = len(dataset)
	train_size = int(dataset_size*0.8)
	val_size = dataset_size - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


	batch = cfg['dataset']['batch'] if args.batch < 0 else args.batch

	devices = select_device(args.device)

	num_workers = int(os.cpu_count() / len(args.device)) if cfg['dataset']['num_workers'] == -1 else cfg['dataset']['num_workers']
	train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=num_workers)
	val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=num_workers)

	##############################
	#       BUILD MODEL          #
	##############################
	resume = None
	if args.resume != '':
		resume = torch.load(args.resume)
	model = FaceBlur(cfg, args, resume)


	##############################
	#       Training SetUp       #
	##############################
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=args.save,
		save_last=True,
		save_top_k=2,
		monitor='val_loss',
		mode='min',
		verbose=True
	)
	callbacks = [checkpoint_callback, TQDMProgressBar()]

	if args.earlystop:
		callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=7))

	if len(devices) > 1:
		trainer = pl.Trainer(
			callbacks=callbacks,
			accelerator="gpu",
			gpus=devices,
			logger=logger,
			max_epochs=cfg['train']['epochs'],
			log_every_n_steps=1,
			plugins=DDPPlugin(find_unused_parameters=False),
			resume_from_checkpoint = args.resume if '.ckpt' in args.resume else None
		)
	
	else:
		trainer = pl.Trainer(
			callbacks=callbacks,
			accelerator="gpu",
			gpus=devices,
			logger=logger,
			max_epochs=cfg['train']['epochs'],
			log_every_n_steps=1,
			resume_from_checkpoint = args.resume if '.ckpt' in args.resume else None
		)


	##############################
	#       START TRAINING !!    #
	##############################
	trainer.fit(model, train_dataloader, val_dataloader)

def select_device(device):
	'''Return device for training'''
	visible_gpu = []
	if isinstance(device, int) and device < 0:
		return 'cpu'
	
	if isinstance(device, list):
		for device_ in device:
			visible_gpu.append(int(device_))
		print(visible_gpu)
	elif isinstance(device, int):
		visible_gpu = [device]
	return visible_gpu

def build_optim(cfg, model):
	'''Return optimizer'''
	optim = cfg['train']['optim']
	optim = optim.lower()
	lr = cfg['train']['lr']

	if optim == 'sgd':
		print("Optimizer :: SGD")
		return torch.optim.SGD(model.parameters(), lr=lr)
	
	if optim == 'adam':
		print("Optimizer :: Adam")
		return torch.optim.Adam(model.parameters(), lr=lr)
	
	# TODO : add optimizer

def build_scheduler(cfg, optim):
	'''Return learning rate scheduler'''
	scheduler_dict = cfg['train']['scheduler']
	scheduler, spec = list(scheduler_dict.items())[0]
	scheduler = scheduler.lower()
	
	if scheduler == 'multisteplr':
		print("Scheduler :: MultiStepLR")
		return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[spec["milestones"]], gamma=spec["decay"])

	if scheduler == 'cycliclr':
		print("Scheduler :: CyclicLR")
		return torch.optim.lr_scheduler.CyclicLR(optim, base_lr=spec["base_lr"], max_lr=spec["max_lr"])
	
	# TODO : add leraning rate scheduler


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='./config/baseline.yaml', help='Path of configuration file')
	parser.add_argument('--save', type=str, default='', help='Path to save the model')
	parser.add_argument('--batch', type=int, default=-1, help='Batch size for training')
	parser.add_argument('--device', type=int, nargs='+', default=-1, help='Specify gpu number for training')
	parser.add_argument('--resume', type=str, default='', help='path to saved model')
	parser.add_argument('--earlystop', action='store_true', help='Whether to proceed with early stopping or not')
	args = parser.parse_args()

	with open(f'config/{args.config}.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	
	train(cfg, args)

	
