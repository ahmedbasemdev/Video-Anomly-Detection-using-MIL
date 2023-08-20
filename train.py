import numpy as np
from model import AnomalyModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import CustomDataset
from utils import MILRankLoss
from tqdm import tqdm
from config import settings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnomalyModel()
model = model.to(device)

bags = np.load("featureExtraction/extractedFeatures/bags.npy")
labels = np.load("featureExtraction/extractedFeatures/labels.npy")
print(f" The Size of Each bag is {bags[0].shape}")
ab_bags = np.load("featureExtraction/extractedFeatures/ab_bags.npy")
ab_labels = np.load("featureExtraction/extractedFeatures/ab_labels.npy")

print(f"Number of Normal Bags is {bags.shape[0]} \n")
print(f"Number of ABNormal Bags is {ab_bags.shape[0]} \n")

dataset = CustomDataset(bags, labels, ab_bags, ab_labels)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = MILRankLoss

print("Start Training ...")
for epoch in range(settings.epochs_number):
    model.train()
    train_loss = 0

    for bags, labels, ab_bags, ab_labels in tqdm(data_loader):
        data = torch.cat((bags, ab_bags), dim=1).to(device)
        batch_size = data.shape[0]
        data = data.view(-1, data.size(-1)).to(device)

        outputs = model(data)
        loss = criterion(outputs, batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'loss = {train_loss / len(data_loader)}')
