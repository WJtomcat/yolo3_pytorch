import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)



class ListDataset(Dataset):
    def __init__(self, list_path, num_classes=80, anchor_path='./config/yolo_anchors.txt', img_size=416,):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_size = img_size
        self.max_objects = 50
        self.num_classes = num_classes

        anchors = get_anchors(anchor_path)
        # self.anchors = np.expand_dims(anchors, 0)
        self.anchors = anchors

        self.num_layers = len(anchors)//3
        self.anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if self.num_layers==3 else [[3,4,5], [0,1,2]]
        img_shape = np.array([img_size, img_size])
        self.grid_shapes = [img_shape//{0:32, 1:16, 2:8}[l] for l in range(self.num_layers)]


    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)

        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape

        input_img = resize(input_img, (self.img_size, self.img_size, 3), mode='reflect')

        input_img = np.transpose(input_img, (2, 0, 1))

        input_img = torch.from_numpy(input_img).float()


        #----------
        # Label
        #----------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)

            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)

            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]

            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w * self.img_size
            labels[:, 4] *= h / padded_h * self.img_size

        y_true = self.preprocess_true_box(labels)

        y_true = [torch.from_numpy(y) for y in y_true]

        return img_path, input_img, y_true

    def preprocess_true_box(self, true_boxes):

        boxes_xy = true_boxes[::, 1:3]
        boxes_wh = true_boxes[::, 3:5]

        # boxes_wh = boxes_wh * self.img_size

        y_true = [np.zeros((self.grid_shapes[l][0], self.grid_shapes[l][1], len(self.anchor_mask[l]), 6+self.num_classes),
            dtype='float32') for l in range(self.num_layers)]

        boxes_wh_expand = np.expand_dims(boxes_wh, 1)
        anchors = np.expand_dims(self.anchors, 0)

        intersect = np.minimum(boxes_wh_expand, anchors)
        intersect = intersect[:,:,0] * intersect[:,:,1]

        union = boxes_wh_expand[:,:,0] * boxes_wh_expand[:,:,1] + anchors[:,:,0] * anchors[:,:,1] - intersect

        anchor_iou = intersect / union

        best_anchor = np.argmax(anchor_iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(self.num_layers):
                if n in self.anchor_mask[l]:

                    i = np.floor(true_boxes[t,1]*self.grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t,2]*self.grid_shapes[l][0]).astype('int32')
                    k = self.anchor_mask[l].index(n)

                    c = true_boxes[t,0].astype('int32')
                    y_true[l][j,i,k,0:2] = true_boxes[t,1:3]
                    y_true[l][j,i,k,2:4] = np.log(true_boxes[t, 3:5] / self.anchors[n] + 1e-16)
                    y_true[l][j,i,k,4] = 1
                    y_true[l][j,i,k,6+c] = 1
                    y_true[l][j,i,:,5] = anchor_iou[t, self.anchor_mask[l]] > 0.7

        return y_true


    def __len__(self):
        return len(self.img_files)
