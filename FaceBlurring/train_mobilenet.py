import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import FaceDataset
from models.mobilenet import FaceMobileNet
import pytorch_model_summary

def log_progress(epoch, num_epoch, iteration, num_data, batch_size, loss):
    progress = int(iteration/(num_data // batch_size)*100//4)
    print(
        f"Epoch : {epoch}/{num_epoch} >>>> train : {iteration}/{num_data // batch_size}{iteration / (num_data//batch_size) * 100:.2f}"
        + '=' * progress + '>' + ' ' * (25 - progress) + f") loss : {loss: .6f}", end='\r')

if __name__ == '__main__':
	device = "cuda" if torch.cuda.is_available() else "cpu"
	batch = 32
	input_size = 1024
	learning_rate = 1e-4
	epochs = 10

	dataset = FaceDataset('./config/datadir.txt/', 'blur', 'degree')
	dataset_size = len(dataset)
	train_size = int(dataset_size*0.8)
	val_size = int(dataset_size*0.1)
	test_size = dataset_size - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	print(f"Training dataset size : {len(train_dataset)}")
	print(f"Validation dataset size : {len(val_dataset)}")
	print(f"Testing dataset size : {len(test_dataset)}")

	train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
	
	model = FaceMobileNet(input_size=input_size).to(device)
	print("Model configuration : ")
	print(pytorch_model_summary.summary(model, torch.zeros(32, 3, input_size, input_size), show_input=True))
	
	criterion = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	history = {"T_loss": [], "V_loss": []}
	os.makedirs("./checkpoint/mobilenet/", exist_ok=True)

	print("Training ... ")
	for epoch in range(epochs):
		model.train()
		training_loss = 0.0
		for i, (image, label) in enumerate(train_dataloader):
			image, label = image.to(device), label.to(device)
			optimizer.zero_grad()
			prediction = model(image)
			loss = criterion(prediction, label)
			training_loss += loss.item()
			loss.backward()
			optimizer.step()
			log_progress(epoch, epochs, i, len(train_dataloader), batch, training_loss/(i+1))
		history["T_loss"].append(training_loss/len(train_dataloader))

		model.eval()
		with torch.no_grad():
			validating_loss = 0.0
			for (image, label) in val_dataloader:
				image, label = image.to(device), label.to(device)
				prediction = model(image)
				loss = criterion(prediction, label)
				validating_loss += loss.item()

			print(f"(Finish) Epoch : {epoch}/{epochs} >>>> Validation loss : {validating_loss/len(val_dataloader):.6f}")
			history["V_loss"].append(validating_loss/len(val_dataloader))

		torch.save(model, f"./checkpoint/mobilenet/checkpoint_{epoch}.pt")