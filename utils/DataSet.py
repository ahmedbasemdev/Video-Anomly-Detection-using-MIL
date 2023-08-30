import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.bags = np.load("featureExtraction/extractedFeatures/bags.npy")
        self.labels = np.load("featureExtraction/extractedFeatures/labels.npy")
        self.ab_bags = np.load("featureExtraction/extractedFeatures/ab_bags.npy")
        self.ab_labels = np.load("featureExtraction/extractedFeatures/ab_labels.npy")

        print(f"There are {self.bags.shape[0]} Normal Bags in Data")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.bags[index], self.labels[index],
                self.ab_bags[index], self.ab_labels[index])
