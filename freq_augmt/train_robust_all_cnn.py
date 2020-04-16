# %%
import torch
from torch import nn
from models import F_AllConvNet
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.autograd import Variable


def generate_normal_mask(scale=2, t=0.5, output_shape=(32, 32)):
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2, ))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal([0, 0], [[1.0, 0], [0, 1.0]])
    # plt.contourf(x, y, 1 - rv.pdf(pos))
    normal_dis = scale * (t - rv.pdf(pos)[None, :])
    normal_dis = torch.tensor(normal_dis).unsqueeze(1)
    normal_dis = torch.nn.functional.adaptive_avg_pool2d(
        normal_dis, output_shape)
    return normal_dis.squeeze(1)


def freq_loss(output, target, alpha=0.8, device='cuda:0'):
    mse_loss = nn.functional.mse_loss(output, target)
    gray_output = torch.mean(output, dim=1)
    spectrum = torch.rfft(gray_output, signal_ndim=2, onesided=False)
    spectrum = torch.norm(spectrum, p='fro', dim=-1)
    mask = generate_normal_mask().to(device)
    shift_loss = torch.mean(spectrum * mask)
    return alpha * mse_loss + (1 - alpha) * shift_loss


batch_size = 16
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')
device = 'cuda:0'

model = F_AllConvNet(3)
model = model.to(device)
mse_criterion = nn.MSELoss()  # mean square error loss
ce_criterion = nn.CrossEntropyLoss()  # mean square error loss
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 10

# %%
best_loss = np.inf
weight_path = 'pre_train.pt'
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, data in enumerate(trainloader):
        img, _ = data
        img = img.to(device)
        recon = model(img, train_fn=True)
        loss = mse_criterion(recon, img)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = epoch_loss / (i + 1)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), weight_path)
        print("Find a better model, save it to " + weight_path)
    print('[Epoch:{}],             MSE Loss:{:.4f}'.format(
        epoch + 1, float(epoch_loss)))

# %%

best_acc = 0
weight_path = 'fall_cnn.pt'
num_epochs = 50
model.load_state_dict(torch.load('pre_train.pt'))

beta = 1.0
alpha = 0.5

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    model.train()
    for i, data in enumerate(trainloader):
        img, labels = data
        img = Variable(img).to(device)
        labels = Variable(labels).to(device).long()
        output, recon = model(img, train_fn=False)
        reconst_loss = freq_loss(recon, img, alpha=alpha)
        ce_loss = ce_criterion(output, labels)
        loss = beta * ce_loss  # + (1 - beta) * reconst_loss
        # print("CE: ", float(ce_loss.data), "recon_loss: ",
        #       float(reconst_loss.data))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.data
        _, predicted = torch.max(nn.functional.softmax(output), 1)
        correct += (predicted.cpu() == labels.cpu()).sum().item()

    epoch_loss = epoch_loss / (i + 1)
    acc = correct / ((i + 1) * batch_size)
    print('[TRAIN] Epoch:{}, Loss:{:.4f}, Acc: {:.3f}'.format(
        epoch + 1, float(epoch_loss), acc))
    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            correct = 0
            for i, data in enumerate(testloader):
                img, labels = data
                img = Variable(img).to(device)
                labels = Variable(labels).to(device).long()
                output, recon = model(img, train_fn=False)
                reconst_loss = freq_loss(recon, img, alpha=0.5)
                ce_loss = ce_criterion(output, labels)
                loss = beta * ce_loss + (1 - beta) * reconst_loss
                epoch_loss += loss.data
                _, predicted = torch.max(nn.functional.softmax(output), 1)
                correct += (predicted.cpu() == labels.cpu()).sum().item()

            epoch_loss = epoch_loss / (i + 1)
            acc = correct / ((i + 1) * batch_size)
            print('[TEST] Epoch:{}, Loss:{:.4f}, Acc: {:.3f}'.format(
                epoch + 1,
                float(epoch_loss),
                acc,
            ))

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), weight_path)
                print("Find a better model, save it to " + weight_path)

# %%
