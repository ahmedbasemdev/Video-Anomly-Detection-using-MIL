import torch
import torch.nn as nn


import torch
import torch.nn.functional as F

lambda1, lambda2 = 8e-5, 8e-5


def MILRankLoss(y_pred, batch_size):
    loss = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    y_pred = y_pred.view(batch_size, -1)

    for i in range(batch_size):
        anomaly_index = torch.randperm(8).cuda()
        normal_index = torch.randperm(8).cuda()

        y_anomaly = y_pred[i, :8][anomaly_index]
        y_normal = y_pred[i, 8:][normal_index]

        y_anomaly_max = torch.max(y_anomaly)  # anomaly

        y_normal_max = torch.max(y_normal)  # normal

        loss += F.relu(1. - y_anomaly_max + y_normal_max)

        sparsity += torch.sum(y_anomaly) * lambda2
        smooth += torch.sum((y_pred[i, :7] - y_pred[i, 1:8]) ** 2) * lambda1

    loss = (loss + sparsity + smooth) / batch_size

    return loss

