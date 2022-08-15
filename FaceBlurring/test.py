import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import FaceDataset
# from models.mobilenet import FaceMobileNet
# import pytorch_model_summary
import yaml
import argparse
from models.model_factory import model_build
import pytorch_lightning as pl

# FIXME : import video_test


# def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss):
#     progress = int(iteration/(num_data // batch_size)*100//4)
#     print(
#         f"Epoch : {epoch}/{num_epoch} >>>> train : {iteration}/{num_data // batch_size}{iteration / (num_data//batch_size) * 100:.2f}"
#         + '=' * progress + '>' + ' ' * (25 - progress) + f") loss : {loss: .6f}", end='\r')



def test(cfg, args, mode):


	##############################
	#       BUILD MODEL          #
	##############################


	model = model_build(model_name=cfg['train']['model'], num_classes=1)
	# only predict blur regression label -> num_classes = 1
	
	if '.ckpt' or '.pt' in args.resume:
		model_state = torch.load(args.resume)
		model = model.load_state_dict(model_state)

	device = args.device
	if 'cuda' in device and torch.cuda.is_available():
		model = model.to(device)
	

	##############################
	#       MODE : VIDEO         #
	##############################


	if mode == 'video':
		raise NotImplementedError()


	##############################
	#       MODE : IMAGE         #
	##############################


	if mode == 'image':
		raise NotImplementedError()
	




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, default='')
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--resume', type=str, default='', required=True)
	parser.add_argument('--mode', type=str)
	args = parser.parse_args()


	with open(args.config, 'r') as f:
		cfg = yaml.safe_load(f)

	mode = args.mode.lower()
	assert mode in ['video', 'image']

	test(cfg, args, mode)

	


	
