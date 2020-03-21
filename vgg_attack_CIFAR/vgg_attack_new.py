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

# set whether to use GPU
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

####################################
# imshow func
####################################
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def _imshow(img):
    imshow(torchvision.utils.make_grid(img))

####################################
# model dir
####################################
MODEL = "model_best.pth.tar"

####################################
# load model
####################################
model = vgg.vgg19()

if os.path.isfile(MODEL):
    print("=> loading model '{}'".format(MODEL))
    checkpoint = torch.load(MODEL)
    print('best_prec: %3f' % checkpoint['best_prec1'])
    model.load_state_dict(checkpoint, strict=False)
    print('model loaded')
else:
    print("=> no checkpoint found at '{}'".format(MODEL))

####################################
# load data
####################################
batch_size = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


[cln_data, true_label] = list(itertools.islice(testloader, 1))[0]
cln_data, true_label = cln_data.to(device), true_label.to(device)


####################################
# Construct a LinfPGDAttack adversary instance
####################################
adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

####################################
#Perform untargeted attack
####################################
adv_untargeted = adversary.perturb(cln_data, true_label)

####################################
# perform targeted attack
####################################
target = torch.ones_like(true_label) * 3
adversary.targeted = True
adv_targeted = adversary.perturb(cln_data, target)

####################################
# Visualization of attacks
####################################

_, pred_cln = torch.max(model(cln_data), 1)
_, pred_untargeted_adv = torch.max(model(adv_untargeted), 1)
_, pred_targeted_adv = torch.max(model(adv_targeted), 1)


plt.figure(figsize=(10, 8))
for ii in range(batch_size):
    plt.subplot(3, batch_size, ii + 1)
    _imshow(cln_data[ii])
    plt.title("clean \n pred: {}".format(classes[pred_cln[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size)
    _imshow(adv_untargeted[ii])
    plt.title("untargeted \n adv \n pred: {}".format(classes[pred_untargeted_adv[ii]]))
    plt.subplot(3, batch_size, ii + 1 + batch_size * 2)
    _imshow(adv_targeted[ii])
    plt.title("targeted to 3 \n adv \n pred: {}".format(classes[pred_targeted_adv[ii]]))

plt.tight_layout()
plt.savefig('test.png')


