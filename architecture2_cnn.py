################################################################################
# CSE 190: Programming Assignment 3
# Fall 2018
# Code author: Udaikaran Singh
#
# Filename: architecture2_cnn.py
# 
# Description: 
# 
# 5 convolution layers -> maxpooling ->
# -> fc1 -> fc2 (output)
#
# Maxpooling is  [8x8] kernel
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


class arch2_cnn(nn.Module):
    def __init__(self):
        super(arch2_cnn, self).__init__()
        
        #convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 20)
        self.conv1_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv1.weight)
        
        #convolution layer 2
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 14, kernel_size = 20)
        self.conv2_normed = nn.BatchNorm2d(14)
        torch_init.xavier_normal_(self.conv2.weight)

        #convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels = 14, out_channels = 12, kernel_size = 20)
        self.conv3_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv3.weight)

        #convolutional layer 4
        self.conv4 = nn.Conv2d(in_channels = 12, out_channels = 10, kernel_size = 20)
        self.conv4_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv4.weight)

        #convolutional layer 5
        self.conv5 = nn.Conv2d(in_channels = 10, out_channels = 8, kernel_size = 20)
        self.conv5_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv5.weight)

        #pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)

        #fully connected layer 1
        self.fc1 = nn.Linear(in_features = 20808, out_features = 128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)

        #fully connected layer 2
        self.fc2 = nn.Linear(in_features = 128, out_features = 14).cuda()
        torch_init.xavier_normal_(self.fc2.weight)


    def forward(self, batch):
        
        #convolutional layers
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = self.pool(batch)
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = func.relu(self.conv5_normed(self.conv5(batch)))
        batch = self.pool(batch)
        
        #flattening data
        batch = batch.view(-1, self.num_flat_features(batch))
        
        #Fully Connected Layers
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        batch = self.fc2(batch)

        return func.sigmoid(batch)


    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features