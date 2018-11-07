# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:06:25 2018

@author: Alex
"""

import torch
import numpy as np
import torchvision.utils as vutils
import torchvision.datasets as dset
from torch.autograd.variable import Variable
import torchvision.transforms as transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import random

class Dnet(torch.nn.Module):
    def __init__(self,ngpu):
        super(Dnet,self).__init__()
        self.ngpu = ngpu
        n_features = 28*28
        n_out = 1
        self.layer1 = nn.Sequential(
                nn.Linear(n_features,1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.layer2 = nn.Sequential(
                nn.Linear(1024,512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.layer3 = nn.Sequential(
                nn.Linear(512,256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out(x)
        return(x)

class Gnet(torch.nn.Module):
    def __init__(self,ngpu):
        super(Gnet,self).__init__()
        self.ngpu = ngpu
        n_features = 100
        n_out = 28*28
        self.layer1 = nn.Sequential(
                nn.Linear(n_features,256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.layer2 = nn.Sequential(
                nn.Linear(256,512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.layer3 = nn.Sequential(
                nn.Linear(512,1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.out = nn.Sequential(
            torch.nn.Linear(1024, n_out),
            torch.nn.Tanh()
        )
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.out(x)
        return(x)
        

def noise(size):
    n = Variable(torch.randn(size,100))
    return n


def img2vec(images):
    return images.view(images.size(0),784)

def vec2img(vectors):
    return vectors.view(vectors.size(0),1,28,28)


manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "C:\\Users\Alex\SGN-26006GAN\GAN_data"
worker = 4
ngpu = 1
size = 28* 28
num_epoch = 20
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

train_set = dset.MNIST(root = dataroot, train = True, transform = trans, download = True)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

num_batches = len(train_loader)


dnet = Dnet(ngpu).to(device)
gnet = Gnet(ngpu).to(device)
d_optimizer = optim.Adam(dnet.parameters(), lr=0.0002)
g_optimizer = optim.Adam(gnet.parameters(), lr=0.0002)
loss = nn.BCELoss()

img_list = []
g_losses = []
d_losses = []
iters = 0
fixed_noise = torch.randn(64,100,device = device);
print("Starting Training Loop........")

for epoch in range(num_epoch):
    for i,data in enumerate(train_loader,0):
        dnet.zero_grad()
        real_img = img2vec(data[0]).to(device)
        batchsize = real_img.size(0)
        label = torch.full((batchsize,),1,device = device)
        output = dnet(real_img).view(-1)
        errD_real = loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        fake_noise = noise(batchsize).to(device)
        
        fake = gnet(fake_noise)
        label.fill_(0)
        output = dnet(fake.detach()).view(-1)
        errD_fake = loss(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
        d_optimizer.step()
        
        
        
        gnet.zero_grad()
        label.fill_(1)
        output = dnet(fake).view(-1)
        errG = loss(output,label)
        errG.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()
        
        
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epoch, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        g_losses.append(errG.item())
        d_losses.append(errD.item())
        
        
    if True:          #((epoch == num_epoch-1) and (i == len(train_loader)-1)):
        with torch.no_grad():
            fake = gnet(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(vec2img(fake),padding = 2,normalize= True))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

