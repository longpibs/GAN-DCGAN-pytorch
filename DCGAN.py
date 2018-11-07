# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:41:29 2018

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
        n_f = 128
        c_in = 1
        c_out = 1
        self.layer1 = nn.Sequential(
                nn.Conv2d(c_in, n_f, 4, 2, 1,bias = False),
                nn.LeakyReLU(0.2),
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(n_f, n_f*2, 4, 2, 1,bias = False),
                nn.BatchNorm2d(n_f*2),
                nn.LeakyReLU(0.2),
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(n_f*2 ,n_f*4 ,4 ,2 ,1 ,bias = False),
                nn.BatchNorm2d(n_f*4),
                nn.LeakyReLU(0.2),
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(n_f*4 ,n_f *8 ,4 ,2 ,1 ,bias = False),
                nn.BatchNorm2d(n_f*8),
                nn.LeakyReLU(0.2),
                )
        self.out = nn.Sequential(
                nn.Conv2d(n_f*8 ,c_out ,4 ,1 ,0 ,bias = False),
                nn.Sigmoid(),
                )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        return(x)

class Gnet(torch.nn.Module):
    def __init__(self,ngpu):
        super(Gnet,self).__init__()
        self.ngpu = ngpu
        n_features = 128
        n_z = 100
        c_out = 1
        self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(n_z,n_features*8,4,1,0,bias = False),
                nn.BatchNorm2d(n_features*8),
                nn.ReLU()
                )
        self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(n_features*8,n_features*4,4,2,1,bias = False),
                nn.BatchNorm2d(n_features*4),
                nn.ReLU()
                )
        self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(n_features*4,n_features*2,4,2,1,bias = False),
                nn.BatchNorm2d(n_features*2),
                nn.ReLU()
                )
        self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(n_features*2,n_features,4,2,1,bias = False),
                nn.BatchNorm2d(n_features),
                nn.ReLU()
                )
        self.out = nn.Sequential(
                nn.ConvTranspose2d(n_features,c_out,4,2,1,bias = False),
                nn.Tanh()
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        return(x)
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "C:\\Users\Alex\SGN-26006GAN\DCGAN_data"
worker = 4
ngpu = 1
size = 64
num_epoch = 20
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

trans = transforms.Compose([transforms.Scale(size),transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

train_set = dset.MNIST(root = dataroot, train = True, transform = trans, download = True)

batch_size = 128

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)

num_batches = len(train_loader)


dnet = Dnet(ngpu).to(device)
gnet = Gnet(ngpu).to(device)
dnet.apply(weights_init)
gnet.apply(weights_init)
print(dnet)
print(gnet)



g_optimizer = optim.Adam(gnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(dnet.parameters(), lr=0.0002, betas=(0.5, 0.999))
loss = nn.BCELoss()

img_list = []
g_losses = []
d_losses = []
iters = 0
fixed_noise = torch.randn(64,100,1,1,device = device);
print("Starting Training Loop........")

for epoch in range(num_epoch):
    img_list.clear()
    for i,data in enumerate(train_loader,0):
        dnet.zero_grad()
        real_img = data[0].to(device)
        batchsize = real_img.size(0)
        label = torch.full((batchsize,),1,device = device)
        output = dnet(real_img).view(-1)
        errD_real = loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        fake_noise = torch.randn(batchsize,100,1,1,device = device);
        
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
        
    with torch.no_grad():
        fake = gnet(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake,padding = 2,normalize= True))
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
