import numpy as np
from utils import MILRankLoss
from model import AnomalyModel
import torch
from torch.utils.data import Dataset, DataLoader
from utils import common, CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnomalyModel()
model.train()
#
# loss_fn = MILRankLoss()
# optimizer = torch.optim.adagrad()
#

bags = np.load("featureExtraction/extractedFeatures/bags.npy")
labels = np.load("featureExtraction/extractedFeatures/labels.npy")

ab_bags = np.load("featureExtraction/extractedFeatures/ab_bags.npy")
ab_labels = np.load("featureExtraction/extractedFeatures/ab_labels.npy")
dataset = CustomDataset(bags, labels, ab_bags, ab_labels)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(common['number_epochs']):

    for bags, labels, ab_bags, ab_labels in data_loader:
        print(model(bags).shape)
        break
    break
