# Created by Samuele Buosi 25/8/18

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from torch.autograd import Variable

from PIL import Image
import os, random
import pandas as pd
import seaborn as sb

import argparse
import Utils as u
import json

argP = argparse.ArgumentParser(description='USAGE: train.py data_dir --arch --hidden_units1 --learning_rate --epochs --dropout --gpu')

argP.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
argP.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
argP.add_argument('--hidden_units1', type=int, dest="fc1_nodes", action="store", default=1000)
argP.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argP.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
argP.add_argument('--gpu', dest="gpu", action="store", default="cpu")
#argP.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpointPy.pth")
argP.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)

args = argP.parse_args()

# Check on input args
if args.arch not in ['vgg16', 'vgg13', 'alexnet', 'resnet18', 'densenet121']:
    print('The specified architecture is not available')

img_path = args.data_dir
arch = args.arch
dropout = args.dropout
fc1_nodes = args.fc1_nodes
learning_rate = args.learning_rate
epochs = args.epochs
gpu = args.gpu
outputNodes = 102
print_every = 20

trainloader, validloader, testloader, image_datasets = u.load_dataSet(img_path)

model, criterion, optimizer  = u.nn_init(arch, dropout, fc1_nodes, outputNodes, learning_rate, gpu)

#u.do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu)

model.class_to_idx = image_datasets['train'].class_to_idx
u.save_checkpoint(img_path, model, optimizer, epochs, fc1_nodes, learning_rate, arch)

print("Model Configured, Trained and Saved!")
