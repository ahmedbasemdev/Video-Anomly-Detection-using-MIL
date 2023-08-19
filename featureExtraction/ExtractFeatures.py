import numpy as np
import torch
import VideoPreprocessing as vp
from pretrained import c3d_model
import os
from glob import glob
from tqdm import tqdm
import json


with open("../utils/config.json") as json_data_file:
    config = json.load(json_data_file)

number_segments = config['number_segments']
number_frames = config['number_frames']

normal_videos = glob("/content/Training-Normal-Videos-Part-1/*")
abnormal_videos = glob("/content/Anomaly-Videos-Part-1/*/*")

data = {"bags": [], "labels": []}

print("\nExtract Normal Videos Representation ")
print("Loading .......\n")
for video in tqdm(normal_videos):
    instances = vp.split_video(video, number_segments, number_frames)
    instances = torch.from_numpy(instances).to(torch.float)
    tensors = []
    for i in range(int(number_frames / 16)):
        instances_splited = instances[:, :, 16 * i:16 * (i + 1), :, :]
        output = c3d_model(instances_splited)
        tensors.append(output)
    stacked = torch.stack(tensors, dim=0)
    average_tensor = torch.mean(stacked, dim=0)
    print(average_tensor.shape)

    data['bags'].append(average_tensor.detach().numpy())
    data['labels'].append(0)

bags = np.array(data['bags'])
labels = np.array(data['labels'])
np.save("extractedFeatures/bags.npy", bags)
np.save("extractedFeatures/labels.npy", labels)

print("\nExtract AbNormal Videos Representation")
print("Loading .......\n")

data = {"bags": [], "labels": []}

for video in tqdm(abnormal_videos):
    instances = vp.split_video(video, number_segments, number_frames)
    instances = torch.from_numpy(instances).to(torch.float)
    tensors = []
    for i in range(int(number_frames / 16)):
        instances_splited = instances[:, :, 16 * i:16 * (i + 1), :, :]
        output = c3d_model(instances_splited)
        tensors.append(output)
    stacked = torch.stack(tensors, dim=0)
    average_tensor = torch.mean(stacked, dim=0)
    print(average_tensor.shape)

    data['bags'].append(average_tensor.detach().numpy())
    data['labels'].append(0)

print("##### Done #####")

bags = np.array(data['bags'])
labels = np.array(data['labels'])
np.save("extractedFeatures/ab_bags.npy", bags)
np.save("extractedFeatures/ab_labels.npy", labels)
