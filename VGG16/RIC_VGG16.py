#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import absolute_import, division
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import math
import torchvision


class ConvNet(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(ConvNet, self).__init__()
        
        self.coords11 = self.generate_coordinates(BATCH_SIZE, 128, 128)
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1, stride=1)
        self.relu = nn.ReLU()
        self.coords12 = self.generate_coordinates(BATCH_SIZE, 128, 128)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.coords21 = self.generate_coordinates(BATCH_SIZE, 64, 64)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, stride=1)
        self.coords22 = self.generate_coordinates(BATCH_SIZE, 64, 64)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, stride=1)
        
        self.coords31 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.coords32 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)       
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        
        self.coords41 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv41 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)
        self.coords42 = self.generate_coordinates(BATCH_SIZE, 16, 16)        
        self.conv42 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)       
        self.conv43 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)
        
        self.coords51 = self.generate_coordinates(BATCH_SIZE, 8, 8) 
        self.conv51 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)            
        self.conv52 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)           
        self.conv53 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)
        
        self.avgpool = nn.AvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 30),
        )
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
    
    
    def generate_coordinates(self, batch_size, input_height, input_width):
        
        coords = torch.zeros(input_height, input_width, 2 * 3 * 3)
        
        parameters = torch.zeros(3)
        parameters[0] = batch_size
        parameters[1] = input_height
        parameters[2] = input_width
        
        center = torch.zeros(2)
        center[0]=torch.sub(torch.div(parameters[1], 2.0), 0.5)
        center[1]=torch.sub(torch.div(parameters[2], 2.0), 0.5)
            
        x_grid = torch.arange(0, parameters[1])
        y_grid = torch.arange(0, parameters[2]) 
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
        
        #coords[:,:,8]=grid_x
        #coords[:,:,9]=grid_y
               
        delta_x = torch.sub(grid_x, center[0])
        delta_y = torch.sub(grid_y, center[1])
        PI = torch.mul(torch.Tensor([math.pi]), 2.0)
        theta=torch.atan2(delta_y, delta_x) % PI[0]
        theta=torch.round(10000.*theta)/10000.
        
        coords[:,:,0]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)
        coords[:,:,1]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0)    
        
        coords[:,:,2]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0)
        coords[:,:,3]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),0.0)
        
        coords[:,:,4]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0)
        coords[:,:,5]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),-1.0)
        
        coords[:,:,6]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),0.0)
        coords[:,:,7]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0)
        
        coords[:,:,10]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),0.0)
        coords[:,:,11]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),-1.0)
        
        coords[:,:,12]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),-1.0)
        coords[:,:,13]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0)
        
        coords[:,:,14]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),-1.0)
        coords[:,:,15]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),0.0)
        
        coords[:,:,16]=torch.add(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
        coords[:,:,17]=torch.add(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),-1.0)
        
        coords=coords.expand(batch_size,-1,-1,-1) 
        coords=coords.permute(0, 3, 1, 2)
        coords = coords.cuda()
        
        return Variable(coords, requires_grad=False)    
    
    
    def forward(self, x):
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords11, weight=self.conv11.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords12, weight=self.conv12.weight, padding=(1,1)) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords21, weight=self.conv21.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords22, weight=self.conv22.weight, padding=(1,1)) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords31, weight=self.conv31.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords32, weight=self.conv32.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords32, weight=self.conv33.weight, padding=(1,1)) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords41, weight=self.conv41.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords42, weight=self.conv42.weight, padding=(1,1)) 
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords42, weight=self.conv43.weight, padding=(1,1)) 
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords51, weight=self.conv51.weight, padding=(1,1))
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords51, weight=self.conv52.weight, padding=(1,1))
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords51, weight=self.conv53.weight, padding=(1,1))
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)

        return x
        
def get_ric_vgg(BATCH_SIZE):
    return ConvNet(BATCH_SIZE)

