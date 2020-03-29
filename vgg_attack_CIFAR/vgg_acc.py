import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import argparse
import itertools
import torch
import torch.nn as nn

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from advertorch.attacks import LinfPGDAttack

import vgg
from vgg import VGG

# set whether to use GPU
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

####################################
# model dir
####################################
if use_cuda:
    MODEL = "model.pkl"
else:
    MODEL = "model.pkl"

####################################
# load model
####################################
if os.path.isfile(MODEL):
    print("=> loading model '{}'".format(MODEL))
    if use_cuda:
        model = torch.load(MODEL)['net']
        model.to(device)
    else:
        model = torch.load(MODEL, map_location=device)['net']
    print('model loaded')
else:
    print("=> no checkpoint found at '{}'".format(MODEL))

####################################
# load data
####################################
batch_size = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

####################################
# Test acc on training dataset
####################################
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in trainloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Training accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



####################################
# Test acc on testing dataset
####################################
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Testing accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
