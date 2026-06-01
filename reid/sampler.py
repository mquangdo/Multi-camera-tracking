import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, time, random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import LambdaLR

class RandomIdentitySampler(Sampler):
    '''
    Trick 3.1 – Random Identity Sampling 
    Each batch contains P identities, each with K samples. Total batch size = P×K.
    '''
    
    def __init__(self, dataset, P, K):
        self.P = P
        self.K = K
        self.label2idx = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.label2idx[label].append(idx)
        self.labels = list(self.label2idx.keys())

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)
        batch = []
        for label in labels:
            idxs = self.label2idx[label].copy()
            if len(idxs) < self.K:
                idxs = (idxs * ((self.K // len(idxs)) + 1))[:self.K]
            else:
                random.shuffle(idxs)
                idxs = idxs[:self.K]
            batch.extend(idxs)
            if len(batch) >= self.P * self.K:
                yield from batch[:self.P * self.K]
                batch = []

    def __len__(self):
        return (len(self.labels) // self.P) * (self.P * self.K)