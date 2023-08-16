import torch
import torch.nn as nn


class MILRankLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, max_anomaly, max_normal):
        pass
