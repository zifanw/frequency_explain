import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import argparse
import itertools
import torch
import torch.nn as nn

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, JacobianSaliencyMapAttack, LBFGSAttack

# 								PGD Attack, 	CW Attack, 			Jacobian Attack,			FGSM Attack


import vgg
from vgg import VGG

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
if use_cuda:
    MODEL = "model.pkl"
else:
    MODEL = "model.pkl"

####################################
# load model
####################################

# Set the model 
model = vgg.vgg19()

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


[cln_data, true_label] = next(iter(testloader))
cln_data, true_label = cln_data.to(device), true_label.to(device)


####################################
# Construct an adversary instance
####################################



adversary_CW = CarliniWagnerL2Attack(model, num_classes = len(classes) , 
	confidence=0, targeted=False, learning_rate=0.01, binary_search_steps=9, max_iterations=10000, 
	abort_early=True, initial_const=0.001, clip_min=0.0, clip_max=1.0, loss_fn=None)

adversary_Jacobian = JacobianSaliencyMapAttack(model, num_classes = len(classes), 
	clip_min=0.0, clip_max=1.0, loss_fn=None, theta=1.0, gamma=1.0, comply_cleverhans=False)

adversary_PGD = LinfPGDAttack(
	model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
	nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
	targeted=False)

adversary_FGSM = LBFGSAttack(model, num_classes = len(classes), batch_size=1, binary_search_steps=9, 
	max_iterations=100, initial_const=0.01, clip_min=0, clip_max=1, loss_fn=None, targeted=False)




# Set the type of attack
adversary = adversary_Jacobian

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

if use_cuda:
	cln_data = cln_data.cpu()
	true_label = true_label.cpu()
	adv_untargeted = adv_untargeted.cpu()
	adv_targeted = adv_targeted.cpu()

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

def DCT(image):
	return dct(dct(image, norm="ortho", axis=0), norm="ortho", axis=1)
def iDCT(image):
	return idct(idct(image, norm="ortho", axis=0), norm="ortho", axis=1)

import numpy as np
from scipy.fftpack import dct

def dct_2d(x):
	return dct(dct(x.T).T)
	
# Now try to plot the frequency distribution of the perturbation



import torch_dct as torch_dct




plt.close()

n_rows = cln_data.shape[0]
n_cols = cln_data.shape[1]

_dct_domain_clean_data = torch.zeros((32,32))
_counter_ = 0
for i in range(n_rows):
	for j in range(n_cols):
		_counter_ += 1
		_dct_domain_clean_data += torch_dct.dct_2d(cln_data[i,j])

_dct_domain_clean_data = _dct_domain_clean_data / _counter_

plt.subplot(1, 2, 1)
plt.imshow(_dct_domain_clean_data, cmap='gray')
plt.colorbar()



n_rows = adv_untargeted.shape[0]
n_cols = adv_untargeted.shape[1]

_dct_domain_ = torch.zeros((32,32))
_counter_ = 0
for i in range(n_rows):
	for j in range(n_cols):
		_counter_ += 1
		_dct_domain_ += torch_dct.dct_2d(adv_untargeted[i,j]) - torch_dct.dct_2d(cln_data[i,j])

_dct_domain_ = _dct_domain_ / _counter_

plt.subplot(1, 2, 2)
plt.imshow(_dct_domain_, cmap='gray')
plt.colorbar()



plt.savefig('cln_data_vs_adv_untargeted.png')










plt.close()

plt.subplot(1, 2, 1)
plt.imshow(_dct_domain_clean_data, cmap='gray')
plt.colorbar()



n_rows = adv_targeted.shape[0]
n_cols = adv_targeted.shape[1]

_dct_domain_ = torch.zeros((32,32))
_counter_ = 0
for i in range(n_rows):
	for j in range(n_cols):
		_counter_ += 1
		_dct_domain_ += torch_dct.dct_2d(adv_targeted[i,j]) - torch_dct.dct_2d(cln_data[i,j])

_dct_domain_ = _dct_domain_ / _counter_

plt.subplot(1, 2, 2)
plt.imshow(_dct_domain_, cmap='gray')
plt.colorbar()



plt.savefig('cln_data_vs_adv_targeted.png')




