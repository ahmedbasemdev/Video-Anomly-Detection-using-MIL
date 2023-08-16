import numpy as np

from utils import MILRankLoss
from model import AnomalyModel
import torch
from torch.utils.data import TensorDataset , DataLoader
from utils import common
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnomalyModel()
model.train()
#
# loss_fn = MILRankLoss()
# optimizer = torch.optim.adagrad()
#

# bags = torch.zeros((1600, 32, 4096))
# labels = [0] * 800 + [1] * 800
# labels = torch.from_numpy(np.array((labels)))
# print(bags.shape , labels.shape)
# dataset = TensorDataset(bags,labels)
# data_loader = DataLoader(dataset, batch_size=60 , shuffle=True)
#
#
# for epoch in range(common['number_epochs']):
#
#     for data, labels in data_loader:
#         print(model(data).squeeze())
#         break
#     break


