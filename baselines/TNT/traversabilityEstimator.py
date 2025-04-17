#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dilation=1, 
                 norm=True,  norm_fn='bn', activation=True, act_fn='leaky_relu', act_slope=0.1, maxpool=False, pool_stride=2, pool_kernel=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation, )
        self.activation = activation
        self.norm = norm
        self.maxpool = False

        if self.activation:
            if act_fn == 'relu':
                self.act_fn = nn.ReLU()
            if act_fn == 'leaky_relu':
                self.act_fn = nn.LeakyReLU(act_slope)
            if act_fn == 'tanh':
                self.act_fn = nn.Tanh()
            if act_fn == 'sigmoid':
                self.act_fn = nn.Sigmoid()

        if norm:
            if norm_fn == 'bn':
                self.norm_fn = nn.BatchNorm2d(out_channels)
            if norm_fn == 'in':
                self.norm_fn = nn.InstanceNorm2d(out_channels)
            if norm_fn == 'ln':
                self.norm_fn = nn.LayerNorm(out_channels)

        if self.maxpool:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=2) 
        
        for modele in self.children():
            if isinstance(modele, nn.Conv2d):
                if act_fn == 'leaky_relu':
                    nn.init.kaiming_normal_(modele.weight, nonlinearity='leaky_relu', mode='fan_in')
                elif act_fn == 'relu':
                    nn.init.kaiming_normal_(modele.weight, nonlinearity='relu', mode='fan_in')
                else:
                    nn.init.kaiming_normal_(modele.weight, mode='fan_in')
        
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm_fn(x)
        if self.activation:
            x = self.act_fn(x)
        if self.maxpool:
            x = self.pool(x)
        
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True, dilation=1, out_padding=0,
                 norm=True,  norm_fn='bn', activation=True, act_fn='leaky_relu', act_slope=0.1, maxunpool=False, pool_stride=2, pool_kernel=2):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, bias=bias, dilation=dilation, output_padding=out_padding)
        self.activation = activation
        self.norm = norm
        self.maxunpool = False

        if self.activation:
            if act_fn == 'relu':
                self.act_fn = nn.ReLU()
            if act_fn == 'leaky_relu':
                self.act_fn = nn.LeakyReLU(act_slope)
            if act_fn == 'tanh':
                self.act_fn = nn.Tanh()
            if act_fn == 'sigmoid':
                self.act_fn = nn.Sigmoid()

        if norm:
            if norm_fn == 'bn':
                self.norm_fn = nn.BatchNorm2d(out_channels)
            if norm_fn == 'in':
                self.norm_fn = nn.InstanceNorm2d(out_channels)
            if norm_fn == 'ln':
                self.norm_fn = nn.LayerNorm(out_channels)

        if self.maxunpool:
            self.pool = nn.MaxUnpool2d(kernel_size=pool_kernel, stride=2) 
        
        for modele in self.children():
            if isinstance(modele, nn.ConvTranspose2d):
                if act_fn == 'leaky_relu':
                    nn.init.kaiming_normal_(modele.weight, nonlinearity='leaky_relu', mode='fan_in')
                elif act_fn == 'relu':
                    nn.init.kaiming_normal_(modele.weight, nonlinearity='relu', mode='fan_in')
                else:
                    nn.init.kaiming_normal_(modele.weight, mode='fan_in')
        
    def forward(self, x):
        x = self.deconv(x)
        if self.norm:
            x = self.norm_fn(x)
        if self.activation:
            x = self.act_fn(x)
        if self.maxunpool:
            x = self.pool(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dilation=1, batch_norm=True, act_fn='leaky_relu'):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias, dilation, batch_norm, activation=False)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding, bias, dilation, batch_norm, activation=False)
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        if act_fn == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.1)
        if act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        if act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
            
    def forward(self, x):
        x = self.conv1(x)
        residual = x.clone()
        x = self.act_fn(x)
        x = self.conv2(x)
        x += residual
        x = self.act_fn(x)
        x += residual
        return x

# Model definition
class TravEstimator(nn.Module):
    def __init__(self, in_channels=1, out_channels=14, embed_dim=2048, depth=4, output_dim=(256, 256), maxpool=True):
        super(TravEstimator, self).__init__()

        #Sample Down
        self.conv1 = ConvBlock(in_channels, out_channels=out_channels*3, kernel_size=5, stride=2, padding=1, bias=True)
        if depth < 2:
            raise ValueError('Depth must be at least 2')
        self.res_blocks = nn.ModuleList([ResBlock(out_channels*3, out_channels*3) for i in range(depth-2)])
        self.conv2 = ConvBlock(out_channels*3, out_channels*4, kernel_size=3, stride=1, padding=1, maxpool=True)
        
        self.deconv1 = DeconvBlock(out_channels*4, out_channels*3, kernel_size=3, stride=1, padding=1)
        self.deconv2 = DeconvBlock(out_channels*3, out_channels*2, kernel_size=3, stride=1, padding=1)
        # self.deconv3 = DeconvBlock(out_channels*2, out_channels*1, kernel_size=3, stride=1, padding=1)
        self.deconv3 = DeconvBlock(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1, act_fn='sigmoid')
        
        self.up = nn.Upsample(size=output_dim, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        x = self.conv1(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # print(x.size())
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        # x = self.deconv4(x)
        x = self.up(x)
        return x

