import numpy as np
import torch
import VideoPreprocessing as vp
from pretrained import c3d_model
import os
from glob import glob
from tqdm import tqdm

normal_videos = glob("/content/Training-Normal-Videos-Part-1/*")[:50]
abnormal_videos =glob("/content/Anomaly-Videos-Part-1/*/*")[:50]

data = {"bags": [], "labels": []}

print("Extract Normal Videos Representation \n")
print("Loading .......\n")
for video in tqdm(normal_videos):
    instances = vp.split_video(video, 8, 16)
    instances = torch.from_numpy(instances).to(torch.float)
    output = c3d_model(instances)
    data['bags'].append(output.detach().numpy())
    data['labels'].append(0)

print("Extract AbNormal Videos Representation \n")
print("Loading .......\n")
for video in tqdm(abnormal_videos):
    instances = vp.split_video(video, 8, 16)
    instances = torch.from_numpy(instances).to(torch.float)
    output = c3d_model(instances)
    data['bags'].append(output.detach().numpy())
    data['labels'].append(1)
print("##### Done #####")

bags = np.array(data['bags'])
labels = np.array(data['labels'])
np.save("bags.npy",bags)
np.save("labels.npy",labels)

