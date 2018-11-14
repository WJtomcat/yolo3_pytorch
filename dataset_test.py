from __future__ import division

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from dataset_new import *

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

anchors = get_anchors('./config/yolo_anchors.txt')


dataloader = torch.utils.data.DataLoader(
    ListDataset('./2012_val.txt'), batch_size=2,
)


for batch_i, (img_path, input_img, y_true) in enumerate(dataloader):
    print(y_true[0][0,5,6,2,:])
    break
