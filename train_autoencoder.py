# %%
import torch
import numpy as np
from torch import nn
import torchvision
import torchvision.transforms as transforms


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_space = nn.Sequential(  # latent space
            nn.Linear(9216, 128, bias=True), nn.ReLU(),
            nn.Linear(128, 9216, bias=True), nn.ReLU())
        self.up_pool = nn.Conv2d(1, 3, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3), nn.ReLU(),
            nn.ConvTranspose2d(256,
                               96,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 3, stride=2, padding=1,
                               output_padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        N, C, H, W = x.size()
        x = x.view((N, -1))
        x = self.latent_space(x)
        x = x.view((N, C, H, W))
        x = self.decoder(x)
        return x


# %%
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

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(torch.max(images))

# %%
device = 'cuda'
model = Autoencoder()
model = model.to(device)
model.train()
criterion = nn.MSELoss()  # mean square error loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 100
outputs = []

best_loss = np.inf
weight_path = 'AutoEncoder.pt'
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, data in enumerate(trainloader):
        img, _ = data
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = epoch_loss / (i + 1)
    if epoch > 30 and epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), weight_path)
    print('Epoch:{}, MSE Loss:{:.4f}'.format(epoch + 1, float(loss)))
