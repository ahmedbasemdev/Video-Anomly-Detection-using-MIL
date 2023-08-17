import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, bags, labels, ab_bags, ab_labels):
        super().__init__()

        self.bags = bags
        self.labels = labels

        self.ab_bags = ab_bags
        self.ab_labels = ab_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.bags[index], self.labels[index],
                self.ab_bags[index], self.ab_labels[index])
