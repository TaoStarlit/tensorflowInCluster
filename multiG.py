# -*- coding:UTF-8 -*-
# multi-G retraining

import os
import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def log(x):
    return torch.log(x + 1e-8)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0, 0)


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            # nn.Linear(128, 128),
            # nn.ReLU(True),
            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            # nn.Linear(128, 128),
            # nn.ReLU(True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.main(x)


# main
isCuda = 1
batchSize = 64

# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
# transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
dataset = datasets.MNIST('./', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=4, drop_last=True)
print('size:{}'.format(dataset.test_data.size()))

# create Generator and Discriminator
feature_dim = 28 * 28
z_dim = 10
lr = 1e-3

D = Discriminator(in_dim=feature_dim, out_dim=1).cuda()

D_solver = optim.Adam(D.parameters(), lr=lr)
BCE = nn.BCELoss()

GList = []
GNum = 3
l_real = Variable(torch.ones([batchSize, 1])).cuda()
l_fake = Variable(torch.zeros([batchSize, 1])).cuda()
newG = 0
for epoch in range(1000):
    # multi-fakes vs real data
    if epoch % 5 == 0:
        if epoch > 0:
            newG = 1
            if len(GList) < GNum:
                GList.append(copy.deepcopy(G))
            else:
                GList.pop(0)
                GList.append(copy.deepcopy(G))

        # new a new generator after a few epoch and push old G into Glist
        G = Generator(in_dim=z_dim, out_dim=feature_dim).cuda()
        G_solver = optim.Adam(G.parameters(), lr=lr)

    for batch_idx, (x_in, _) in enumerate(dataloader):
        len_x = x_in.size()[0]
        # Sample data
        z_in = Variable(torch.randn(len_x, z_dim)).cuda()
        x_in = Variable(x_in).cuda()


        # Update GAN D network
        D.zero_grad()

        # Loss function of D for real
        D_real = D(x_in.view(-1, feature_dim))
        errD_real = BCE(torch.sigmoid(D_real), l_real)

        # Loss function of D for multi-fakes
        errD_fake = 0

        if False:
            for mG in GList:
                mG_sample = mG(z_in)
                mD_fake = D(mG_sample)
                errD_fake = errD_fake + BCE(torch.sigmoid(mD_fake), Variable(torch.ones(1)))

        G_sample = G(z_in).detach()
        D_fake = D(G_sample)
        errD_fake = errD_fake + BCE(torch.sigmoid(D_fake), l_fake)

        # errD_fake.backward()
        D_loss = errD_real + errD_fake / (len(GList) + 1)
        D_loss.backward()
        D_solver.step()

        if newG != 0:
            newG = 0
            z_in = Variable(torch.randn(len_x, z_dim)).cuda()
            for ii in range(30):
                # Update GAN G network
                G.zero_grad()
                # z_in = Variable(torch.randn(len_x, z_dim)).cuda()
                G_sample = G(z_in)
                D_fake = D(G_sample)

                # Loss function for Generator
                G_loss = BCE(torch.sigmoid(D_fake), l_real)
                G_loss.backward()
                G_solver.step()
                print('Train new G,{}, G_loss: {:.4},D_fake:{:.4}'.format(ii, G_loss.data[0],D_fake.data[0]))

        G.zero_grad()
        z_in = Variable(torch.randn(len_x, z_dim)).cuda()
        G_sample = G(z_in)
        D_fake = D(G_sample)

        # Loss function for Generator
        G_loss = BCE(torch.sigmoid(D_fake), l_real)
        G_loss.backward()
        G_solver.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t D_loss: {:.4}; G_loss: {:.4}'.format(
                epoch, batch_idx * len(x_in), len(dataloader.dataset), 100 * batch_idx / len(dataloader),
                D_loss.data[0], G_loss.data[0]))

        # Print and plot every now and then
        if epoch % 10 == 0:
            i = 0
            for mG in GList:
                i = i + 1
                mG_sample = mG(z_in).cpu()
                save_image(mG_sample.data.view(len_x, 1, 28, 28), './fig/mG%d_%03d.png' % (i, epoch))
            sample = G(z_in).cpu()
            save_image(sample.data.view(len_x, 1, 28, 28), './fig/G_%03d.png' % epoch)
