# %%
import torch
import numpy as np
from torch import nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # like the Composition layer you built
            nn.Conv2d(3, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            # nn.Conv2d(32, 64, 5, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, 7, stride=2, padding=1),
            # nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_space = nn.Sequential(  # latent space
            nn.Linear(512, 512, bias=True), nn.ReLU())
        # nn.Linear(128, 512, bias=True), nn.ReLU())
        # self.up_pool = nn.Conv2d(1, 3, 1)
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(
            #     128,
            #     64,
            #     7,
            #     stride=2,
            #     padding=1,
            # ), nn.ReLU(),
            # nn.ConvTranspose2d(64,
            #                    32,
            #                    5,
            #                    stride=2,
            #                    padding=1,
            #                    output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128,
                               3,
                               3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        # N, C, H, W = x.size()
        # x = x.view((N, -1))
        # x = self.latent_space(x)
        # x = x.view((N, C, H, W))
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
device = 'cuda'
model = Autoencoder()
criterion = nn.MSELoss()  # mean square error loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
num_epochs = 100
# %%

model = model.to(device)
model.train()
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
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), weight_path)
        print("Find a better model, save it to " + weight_path)
    print('[Epoch:{}],             MSE Loss:{:.4f}'.format(
        epoch + 1, float(epoch_loss)))
# %%
weight_path = 'AutoEncoder.pt'
device = 'cuda'
model = Autoencoder()
model = model.to(device)
model.eval()
model.load_state_dict(torch.load(weight_path))
epoch_loss = 0
test_results = None
with torch.no_grad():
    for i, data in enumerate(testloader):
        img, _ = data
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        epoch_loss += loss.data
        if i == 1:
            test_results = [img.cpu().numpy(), recon.cpu().numpy()]
    epoch_loss = epoch_loss / (i + 1)
    print('Test MSE Loss: ', epoch_loss)

# %%
# %%

inputs, recon = test_results[0][5:9], test_results[1][5:9]

inputs = np.transpose(inputs, (0, 2, 3, 1))
recon = np.transpose(recon, (0, 2, 3, 1))
num_images = inputs.shape[0]
fig, ax = plt.subplots(2, num_images)
for i in range(num_images):
    ax[0, i].imshow(inputs[i])
    ax[1, i].imshow(recon[i])
plt.show()

# %%
output = torch.tensor(test_results[0][5:9])
gray_output = torch.mean(output, dim=1)
spectrum = torch.rfft(gray_output, signal_ndim=2, onesided=False)
spectrum = torch.norm(spectrum, p='fro', dim=-1)

# %%
spectrum = spectrum.numpy()
print(spectrum.shape)
plt.imshow(np.log(np.fft.fftshift(spectrum[0])))

# %%

# %%

# %%
normal_dis.size()

# %%
