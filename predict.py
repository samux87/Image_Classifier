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

argP = argparse.ArgumentParser(description='USAGE: predict.py image_location checkpoint --top_k')

argP.add_argument('image_path', default='/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
argP.add_argument('checkpoint', default='/home/workspace/aipnd-project/checkpointpy.pth', nargs='*', action="store", type = str)
argP.add_argument('--topk', default=5, dest="topk", action="store", type=int)

args = argP.parse_args()

# Load Checkpoint
model, optimizer, epochs = u.load_checkpoint(args.checkpoint)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
probs, classes = u.predict(args.image_path, model, args.topk)

# Get the labels
labels = []
for i in range(len(classes)):
    labels.append(cat_to_name[classes[i]])

print('File path ' + args.image_path)
print('\nMost likely classification: ' + (labels[0]).title() + ' (' + str(round(probs[0]*100)) + '%)\n')



