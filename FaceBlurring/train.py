import os
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision

from dataset.dataset import FaceDataset
from models.model_factory import model_build


# TODO : Pytorch Lightning Wrapping -> Multi GPU
def train(cfg, args):
	'''Function for training face blur detection model'''

	##############################
	#       DataLoader           #
	##############################
	dataset = FaceDataset(
		cfg['dataset']['txt_path'], 
		transform=torchvision.transforms.Compose([torchvision.ToTensor()])
	)		# FIXME
	
	dataset_size = len(dataset)
	train_size = int(dataset_size*0.8)
	val_size = dataset_size - train_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	print(f"Training dataset size : {len(train_dataset)}")
	print(f"Validation dataset size : {len(val_dataset)}")

	batch = cfg['dataset']['batch'] if args.batch < 0 else args.batch

	train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=cfg['dataset']['num_workers'])
	val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=cfg['dataset']['num_workers'])

	##############################
	#       BUILD MODEL          #
	##############################
	model = model_build(model_name=cfg['train']['model'], num_classes=1)
	# only predict blur regression label -> num_classes = 1

	##############################
	#       Training SetUp       #
	##############################
	# loss / optim / scheduler / ...
	loss_func = build_loss_func(cfg)
	optimizer = build_optim(cfg, model)
	scheduler = build_scheduler(cfg, optimizer)
	epochs = cfg['train']['epochs']
	val_epoch = cfg['train']['val_epoch']

	device = args.devices
	if 'cuda' in device and torch.cuda.is_available():
		model = model.to(device)

	os.makedirs(args.save, exist_ok=True)
	history = {"T_loss": [], "V_loss": []}

	# Continue previous training
	if '.ckpt' in args.resume:
		checkpoint = torch.load(args.resume)
		model = model.load_state_dict(checkpoint['model_state_dict'])
		optimizer = optimizer.load_state_dict(checkpoint['optimizers_state_dict'])
		scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

	##############################
	#       START TRAINING !!    #
	##############################
	for epoch in range(epochs):
		model.train()
		training_loss = 0.0
		for i, (image, label) in enumerate(train_dataloader):
			image, label = image.to(device), label.to(device)
			optimizer.zero_grad()
			prediction = model(image)
			loss = loss_func(prediction, label)
			training_loss += loss.item()
			loss.backward()
			optimizer.step()
			if i % (len(train_dataloader)//10) == 0:
				print(f"Epoch #{epoch} [{i}/{len(train_dataloader)}] >>>>>>>> Training loss : {training_loss/(i+1):.6f}")
		history["T_loss"].append(training_loss/len(train_dataloader))

		if epoch and epoch % val_epoch == 0:
			model.eval()
			with torch.no_grad():
				validating_loss = 0.0
				for (image, label) in val_dataloader:
					image, label = image.to(device), label.to(device)
					prediction = model(image)
					loss = loss_func(prediction, label)
					validating_loss += loss.item()
				print(f"(Finish) Epoch : {epoch}/{epochs} >>>> Validation loss : {validating_loss/len(val_dataloader):.6f}")
				history["V_loss"].append(validating_loss/len(val_dataloader))

			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"epoch": epoch
				}
				, f"{args.save}/checkpoint_{epoch}.ckpt")

def build_loss_func(cfg):
	'''Return loss function'''
	for loss_name, weight in cfg['train']['loss']:
		pass
	
	# raise NotImplementedError()
	return None

def build_optim(cfg, model):
	'''Return optimizer'''
	optim = cfg['train']['optim']
	optim = optim.lower()
	lr = cfg['train']['lr']

	if optim == 'sgd':
		return torch.optim.SGD(model.parameters(), lr=lr)
	
	if optim == 'adam':
		return torch.optim.Adam(model.parameters(), lr=lr)
	
	# TODO : add optimizer

def build_scheduler(cfg, optim):
	'''Return learning rate scheduler'''
	scheduler_dict = cfg['train']['scheduler']
	scheduler, spec = scheduler_dict.items()
	scheduler = scheduler.lower()
	
	if scheduler == 'multistep':
		return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[spec["milestones"]], gamma=spec["decay"])

	if scheduler == 'cyclic':
		return torch.optim.lr_scheduler.CyclicLR(optim, base_lr=spec["base_lr"], max_lr=spec["max_lr"])
	
	# TODO : add leraning rate scheduler


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='./config/baseline.yaml', help='Path of configuration file')
	parser.add_argument('--save', type=str, default='', help='Path to save the model')
	parser.add_argument('--batch', type=int, default=-1, help='Batch size for training')
	parser.add_argument('--device', type=str, default='cpu', help='Device for training. It can be "cpu" or "cuda"')
	parser.add_argument('--resume', type=str, default='', help='path to saved model')
	args = parser.parse_args()

	with open(args.config, 'r') as f:
		cfg = yaml.safe_load(f)
	
	train(cfg, args)
