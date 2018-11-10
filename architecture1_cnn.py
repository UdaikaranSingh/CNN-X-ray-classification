################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Udaikaran Singh
#
# Filename: architecture1_cnn.py
# 
# Description: 
# 
# 3 convolution layers -> maxpooling -> sigmoid layer 
# -> fc1 -> fc2 -> fc3 (output)
#
# Maxpooling is  [4x4] kernel
################################################################################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os


class arch1_cnn(nn.Module):
    def __init__(self):
        super(arch1_cnn, self).__init__()
        
        #convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 8)
        self.conv1_normed = nn.BatchNorm2d(4)
        torch_init.xavier_normal_(self.conv1.weight)
        
        #convolution layer 2
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 8)
        self.conv2_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv2.weight)

        #convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 12, kernel_size = 8)
        self.conv3_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv3.weight)
        
        #max-pooling
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = 4)

        #fully connected layer 1
        self.fc1 = nn.Linear(in_features = 178608, out_features = 512)
        self.fc1_normed = nn.BatchNorm1d(512)
        torch_init.xavier_normal_(self.fc1.weight)

        #fully connected layer 2
        self.fc2 = nn.Linear(in_features = 512, out_features = 128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)

        #fully connected layer 3
        self.fc3 = nn.Linear(in_features = 128, out_features = 14).cuda()
        torch_init.xavier_normal_(self.fc3.weight)



    def forward(self, batch):
        
        #convolutional layers
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))

        #pooling
        batch = self.pool(batch)

        #flattening data
        batch = batch.view(-1, self.num_flat_features(batch))

        #fully connected layers
        batch = func.relu(self.fc1(batch))
        batch = func.relu(self.fc2(batch))
        batch = self.fc3(batch)

        return func.sigmoid(batch)



    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features
