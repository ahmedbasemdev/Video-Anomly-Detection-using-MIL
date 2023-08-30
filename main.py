import numpy as np
from model import AnomalyModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import CustomDataset
from utils import MILRankLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from config import AppConfig
from engine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
settings = AppConfig()

model = AnomalyModel()
model = model.to(device)

train_dataset = CustomDataset()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
criterion = MILRankLoss

print("Start Training ...")
for epoch in range(settings.epochs_number):
    loss, current_lr = train(model, train_loader, criterion, optimizer, scheduler, device)
    print(f"Epoch loss is {loss} , LR {current_lr}")
