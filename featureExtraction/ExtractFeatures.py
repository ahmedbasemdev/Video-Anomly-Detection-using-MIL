import numpy as np
import torch
import VideoPreprocessing as vp
from pretrained import c3d_model
import os
from glob import glob
from tqdm import tqdm

normal_videos = glob("../Training-Normal-Videos/*")
abnormal_videos =glob("../Anomaly-Videos/*")
data = {"bags": [], "labels": []}

for video in tqdm(normal_videos):
    instances = vp.split_video(video, 8, 16)
    instances = torch.from_numpy(instances).to(torch.float)
    output = c3d_model(instances)
    data['bags'].append(output.detach().numpy())
    data['labels'].append(0)

for video in tqdm(abnormal_videos):
    instances = vp.split_video(video, 8, 16)
    instances = torch.from_numpy(instances).to(torch.float)
    output = c3d_model(instances)
    data['bags'].append(output.detach().numpy())
    data['labels'].append(1)


bags = np.array(data['bags'])
labels = np.array(data['labels'])
np.save("bags.npy",bags)
np.save("labels.npy",labels)