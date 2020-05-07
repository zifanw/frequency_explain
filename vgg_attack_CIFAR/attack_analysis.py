
# coding: utf-8

# # Table of Contents
#  <p>

# In[1]:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


# import vgg
# from vgg import VGG
from scipy.fftpack import dct, idct

from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model

# set whether to use GPU
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D



# In[2]:


def DCT(image):
	return dct(dct(image, norm="ortho", axis=0), norm="ortho", axis=1)
def iDCT(image):
	return idct(idct(image, norm="ortho", axis=0), norm="ortho", axis=1)

class MMD_loss(nn.Module):
	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
		super(MMD_loss, self).__init__()
		self.kernel_num = kernel_num
		self.kernel_mul = kernel_mul
		self.fix_sigma = None
		return
    
	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
		n_samples = int(source.size()[0])+int(target.size()[0])
		total = torch.cat([source, target], dim=0)
		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
		L2_distance = ((total0-total1)**2).sum(2) 
        
		if fix_sigma:
			bandwidth = fix_sigma
		else:
			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            
		bandwidth /= kernel_mul ** (kernel_num // 2)
		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
		return sum(kernel_val)

	def forward(self, source, target):
		batch_size = int(source.size()[0])
		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
		XX = kernels[:batch_size, :batch_size]
		YY = kernels[batch_size:, batch_size:]
		XY = kernels[:batch_size, batch_size:]
		YX = kernels[batch_size:, :batch_size]
		loss = torch.mean(XX + YY - XY -YX)
		return loss
    
def lscale01(M):

	New_M = np.zeros((M.shape))
	MIN = np.min(M)
	MAX = np.max(M)
	if (MAX == MIN):
		New_M[:, :] = 0.0 * M
	else:
		New_M[:, :] = (M - MIN) / (MAX - MIN)

	return New_M


# In[3]:



####################################
# imshow func
####################################
def imshow(img):
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

def _imshow(img):
	imshow(torchvision.utils.make_grid(img))

'''
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
'''


MODEL = "cifar_nat.pt"

ds = CIFAR('data/cifar-10-batches-py')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path=MODEL, parallel=False)
model = model.model
model.eval()


####################################
# load data
####################################
batch_size = 200

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

# In[4]:


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


# In[22]:


# Set the type of attack

loss = MMD_loss()
adversary = adversary_CW
FOLDER = 'adversary_CW/'
filetxt = FOLDER+'output.txt'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

true = np.zeros((0)).astype(int)
pred_cln = np.zeros((0)).astype(int)
pred_untargeted_adv = np.zeros((0)).astype(int)
pred_targeted_adv = np.zeros((0)).astype(int)

MMD_untargeted = np.zeros((0))
MMD_targeted = np.zeros((0))


fold = 0
DCT_untargeted = 0
DCT_targeted = 0
    
for cln_data, true_label in testloader:
    cln_data, true_label = cln_data.to(device), true_label.to(device)   
    true = np.concatenate((true, true_label.cpu().numpy().astype(int)), axis=None)
    
    _, pred_cln_ = torch.max(model(cln_data), 1)
    pred_cln = np.concatenate((pred_cln, pred_cln_.cpu().numpy().astype(int)), axis=None)    
    
    ####################################
    #Perform untargeted attack
    ####################################
    adv_untargeted = adversary.perturb(cln_data, true_label)

    _, pred_untargeted_adv_ = torch.max(model(adv_untargeted), 1)
    pred_untargeted_adv = np.concatenate((pred_untargeted_adv, pred_untargeted_adv_.cpu().numpy().astype(int)), axis=None)
    
    ####################################
    # perform targeted attack
    ####################################
    target = torch.ones_like(true_label) * 3
    adversary.targeted = True
    adv_targeted = adversary.perturb(cln_data, target)
    
    _, pred_adv_targeted_ = torch.max(model(adv_targeted), 1)
    pred_targeted_adv = np.concatenate((pred_targeted_adv, pred_adv_targeted_.cpu().numpy().astype(int)), axis=None)

    for i in range(len(true_label)):
        image = np.transpose(adv_untargeted[i].cpu().numpy(), (1, 2, 0))
        dct_adv_untargeted = ((DCT(image[:,:,0]) + DCT(image[:,:,1]) + DCT(image[:,:,2]))/3.0)

        image = np.transpose(adv_targeted[i].cpu().numpy(), (1, 2, 0))
        dct_adv_targeted = ((DCT(image[:,:,0]) + DCT(image[:,:,1]) + DCT(image[:,:,2]))/3.0)

        image = np.transpose(cln_data[i].cpu().numpy(), (1, 2, 0))
        dct_image = ((DCT(image[:,:,0]) + DCT(image[:,:,1]) + DCT(image[:,:,2]))/3.0)

        DCT_untargeted = DCT_untargeted + abs((dct_image - dct_adv_untargeted)/dct_image)
        DCT_targeted = DCT_targeted + abs((dct_image - dct_adv_targeted)/dct_image)

        MMD_targeted = np.append(MMD_targeted,loss(torch.from_numpy(dct_adv_targeted), torch.from_numpy(dct_image)).cpu().item())
        MMD_untargeted = np.append(MMD_untargeted,loss(torch.from_numpy(dct_adv_untargeted), torch.from_numpy(dct_image)).cpu().item())
    fold += 1
    if(fold > 10):
        break
        
####################################
# Visualization of attacks
####################################

acc_adv_untargeted = 100*np.mean(true == pred_untargeted_adv)
acc_adv_targeted = 100*np.mean(true == pred_targeted_adv)
acc_cln = 100*np.mean(true == pred_cln)

print('Total Accuracy: ',"Unattacked: ", acc_cln, 
      '\t and during attack: ',FOLDER,'\t',
      
      "Untargeted attack: ",acc_adv_untargeted,
      "\tTargeted attack: ",acc_adv_targeted,
     file=open(filetxt, "a"))

correct_pred_indx = (true == pred_cln)
acc_adv_untargeted = 100*np.mean(true[correct_pred_indx] == pred_untargeted_adv[correct_pred_indx])
acc_adv_targeted = 100*np.mean(true[correct_pred_indx] == pred_targeted_adv[correct_pred_indx])
print('Effect of attack on accurately predcited image by unattacked model:\t'      
      "Untargeted attack: ",acc_adv_untargeted,
      "\tTargeted attack: ",acc_adv_targeted,
     file=open(filetxt, "a"))

DCT_untargeted = (DCT_untargeted/len(true))
DCT_targeted = (DCT_targeted/len(true))

fig, ax = plt.subplots(1,1,figsize=(6, 6))
im1 = ax.imshow(lscale01(DCT_untargeted), cmap='YlOrRd')
ax.title.set_text('untargeted')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)    
fig.colorbar(im1, cax=cax)
# ax.set_axis_off()
plt.savefig(FOLDER+'correct_DCT_untargeted.png')
# plt.show()

fig, ax = plt.subplots(1,1,figsize=(6, 6))
im1 = ax.imshow(lscale01(DCT_targeted), cmap='YlOrRd')
ax.title.set_text('targeted')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)    
fig.colorbar(im1, cax=cax)
# ax.set_axis_off()
plt.savefig(FOLDER+'correct_DCT_targeted.png')
# plt.show()


c_indx_untargeted = np.logical_and(true == pred_cln,true == pred_untargeted_adv)
c_indx_targeted = np.logical_and(true == pred_cln,true == pred_targeted_adv)

true_pred_untargeted = np.sum(c_indx_untargeted)/np.sum(correct_pred_indx)
true_pred_targeted = np.sum(c_indx_targeted)/np.sum(correct_pred_indx)

w_indx_untargeted = np.logical_and(true == pred_cln,true != pred_untargeted_adv)
w_indx_targeted = np.logical_and(true == pred_cln,true != pred_targeted_adv)

false_pred_untargeted = np.sum(w_indx_untargeted)/np.sum(correct_pred_indx)
false_pred_targeted = np.sum(w_indx_targeted)/np.sum(correct_pred_indx)

print("True pred accuracy of samples: ",100*true_pred_untargeted,100*true_pred_targeted,
     file=open(filetxt, "a"))
print("wrong pred accuracy of samples: ",100*false_pred_untargeted,100*false_pred_targeted,
     file=open(filetxt, "a"))

print("Total mean MMD: ", np.mean(MMD_untargeted), 
      np.mean(MMD_targeted),
     file=open(filetxt, "a"))
print("Correct mean MMD: ", np.mean(MMD_untargeted[c_indx_untargeted]), 
      np.mean(MMD_targeted[c_indx_targeted]),
     file=open(filetxt, "a"))
print("wrong mean MMD: ", np.mean(MMD_untargeted[w_indx_untargeted == 0]), 
      np.mean(MMD_targeted[w_indx_targeted == 0]),
     file=open(filetxt, "a"))

fig, ax = plt.subplots(1,1,figsize=(6, 6))
plt.hist(MMD_untargeted,bins=100, weights=100*np.ones(len(MMD_untargeted)) / len(MMD_untargeted))
plt.xlabel("MMD_untargeted")
plt.ylabel("% of images")
plt.title('Histogram of MMD_untargeted')
plt.axvline(np.mean(MMD_untargeted), color='k', linestyle='dashed', linewidth=1)
plt.savefig(FOLDER+'hist_untargeted.png')
# plt.show()

fig, ax = plt.subplots(1,1,figsize=(6, 6))
plt.hist(MMD_targeted,bins=100, weights=100*np.ones(len(MMD_targeted)) / len(MMD_targeted))
plt.xlabel("MMD_targeted")
plt.ylabel("% of images")
plt.title('Histogram of MMD_targeted')
plt.axvline(np.mean(MMD_untargeted), color='k', linestyle='dashed', linewidth=1)
plt.savefig(FOLDER+'hist_targeted.png')
# plt.show()


# In[183]:


if use_cuda:
	cln_data = cln_data.cpu()
	true_label = true_label.cpu()
	adv_untargeted = adv_untargeted.cpu()
	adv_targeted = adv_targeted.cpu()
    

plt.figure(figsize=(10, 8))
num_plots = 8
for ii in range(num_plots):
	plt.subplot(3, num_plots, ii + 1)
	_imshow(cln_data[ii])
	plt.title("clean \n pred: {}".format(classes[pred_cln[ii]]))
	plt.subplot(3, num_plots, ii + 1 + num_plots)
	_imshow(adv_untargeted[ii])
	plt.title("untargeted \n adv \n pred: {}".format(classes[pred_untargeted_adv[ii]]))
	plt.subplot(3, num_plots, ii + 1 + num_plots * 2)
	_imshow(adv_targeted[ii])
	plt.title("target cat \n adv \n pred: {}".format(classes[pred_targeted_adv[ii]]))

plt.tight_layout()
plt.savefig(FOLDER+'test.png')

