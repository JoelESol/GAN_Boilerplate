import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from utils.batchinstancenorm import BatchInstanceNorm2d as Normlayer
import functools
from functools import partial
import torchvision.transforms as ttransforms


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        bin = functools.partial(Normlayer, affine=True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            bin(filters)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                bin(filters)
            )

    def forward(self, x):
        output = self.main(x)
        output += self.shortcut(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        output=self.encoder(x)
        output=self.res_blocks(output)
        output=self.decoder(output)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1, bias=True)),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(512, 16, kernel_size=4,  stride=1, padding=1)),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(3136,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class VGG16_FirstLayer(nn.Module):
    def __init__(self):
        super(VGG16_FirstLayer, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.first_conv = self.features[0]  # Access the first convolutional layer

    def forward(self, x):
        x = self.first_conv(x)  # Pass the input through the first layer
        return x

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def freeze_layers(self):
        for name, param in self.named_parameters():
            if 'classifier' not in name and 'features.28' not in name:
                param.requires_grad = False

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(32768, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.to_relu_1_1 = nn.Sequential()
        self.to_relu_2_1 = nn.Sequential()
        self.to_relu_3_1 = nn.Sequential()
        self.to_relu_4_1 = nn.Sequential()
        self.to_relu_4_2 = nn.Sequential()

        for x in range(2):
            self.to_relu_1_1.add_module(str(x), features[x])
        for x in range(2,7):
            self.to_relu_2_1.add_module(str(x), features[x])
        for x in range(7,12):
            self.to_relu_3_1.add_module(str(x), features[x])
        for x in range(12,21):
            self.to_relu_4_1.add_module(str(x), features[x])
        for x in range(21,25):
            self.to_relu_4_2.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_1(x)
        h_relu_1_1 = h
        h = self.to_relu_2_1(h)
        h_relu_2_1 = h
        h = self.to_relu_3_1(h)
        h_relu_3_1 = h
        h = self.to_relu_4_1(h)
        h_relu_4_1 = h
        h = self.to_relu_4_2(h)
        h_relu_4_2 = h
        out = (h_relu_1_1, h_relu_2_1, h_relu_3_1, h_relu_4_1, h_relu_4_2)
        return out