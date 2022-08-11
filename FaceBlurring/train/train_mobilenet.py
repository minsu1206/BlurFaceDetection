import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from dataset.dataset import FaceDataset
from models.mobilenet import FaceMobileNetV1, FaceMobileNetV2
import torchvision.transforms as transforms
import pytorch_model_summary

if __name__ == '__main__':
	# Hyperparameters
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print("Your current device is :", device)
	batch = 64
	input_size = 256
	learning_rate = 1e-3
	epochs = 10

	# Getting dataset
	print("Geting dataset ...")
	transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)])
	dataset = FaceDataset('../config/datadir.txt', 'blur', 'degree', transform, input_size)
	dataset_size = len(dataset)
	train_size = int(dataset_size*0.8)
	val_size = int(dataset_size*0.1)
	test_size = dataset_size - train_size - val_size

	# Split dataset into train, validation, test
	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	# Check number of each dataset size
	print(f"Training dataset size : {len(train_dataset)}")
	print(f"Validation dataset size : {len(val_dataset)}")
	print(f"Testing dataset size : {len(test_dataset)}")

	# Dataloaders
	train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
	iter_length = len(train_dataloader)

	# Instantiate model configuration
	model = FaceMobileNetV2(input_size=input_size)
	print("Model configuration : ")
	print(pytorch_model_summary.summary(model, torch.zeros(batch, 3, input_size, input_size), show_input=True))

	model.to(device)
	# Criterion, Optimizer, Loss history tracker
	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	history = {"T_loss": [], "V_loss": []}

	# Create directory to save checkpoints
	os.makedirs("./checkpoint/mobilenet/", exist_ok=True)

	# Train model
	writer = SummaryWriter()
	print("Training ... ")
	for epoch in range(epochs):
		model.train()
		training_loss = 0.0
		for i, (image, label) in enumerate(train_dataloader):
			image, label = image.to(device), label.to(device)
			optimizer.zero_grad()
			prediction = model(image)
			loss = criterion(prediction, label.view(-1, 1))
			training_loss += loss.item()
			loss.backward()
			optimizer.step()

			if i % (iter_length//10) == 0:
				print(f"Epoch #{epoch}[{i}/{len(train_dataloader)}] >>>> Training loss : {training_loss/(i+1):.6f}")
		#writer.add_scalar('Loss/train', training_loss/len(train_dataloader), epoch)
		history["T_loss"].append(training_loss/iter_length)
		scheduler.step()

		model.eval()
		with torch.no_grad():
			validating_loss = 0.0
			for (image, label) in val_dataloader:
				image, label = image.to(device), label.to(device)
				prediction = model(image)
				loss = criterion(prediction, label.view(-1, 1))
				validating_loss += loss.item()

			print(f"(Finish) Epoch : {epoch}/{epochs} >>>> Validation loss : {validating_loss/len(val_dataloader):.6f}")
			history["V_loss"].append(validating_loss/len(val_dataloader))
			#writer.add_scalar('Loss/val', training_loss / len(train_dataloader), epoch)

		torch.save(model, f"./checkpoint/mobilenet/checkpoint_{epoch}.pt")

	plt.figure(figsize=(10, 16))
	plt.subplot(2, 1, 1)
	plt.plot(np.arange(epochs), history["T_loss"])
	plt.title("Training loss", fontsize=15)
	plt.subplot(2, 1, 2)
	plt.plot(np.arange(epochs), history["V_loss"])
	plt.title("Validation loss", fontsize=15)
	plt.savefig("loss_hist.png")
