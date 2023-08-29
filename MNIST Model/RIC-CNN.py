#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division
#import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import math

class Rot_Inv_ConvNet(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(Rot_Inv_ConvNet, self).__init__()              
        # conv11
        self.coords11 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv11 = nn.Conv2d(1,32,3,padding=1,stride=1)
        self.bn11 = nn.BatchNorm2d(32)
        # conv12
        self.coords12 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv12 = nn.Conv2d(32,32,3,padding=1,stride=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.mp12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
        # conv21
        self.coords21 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv21 = nn.Conv2d(32,64,3,padding=1,stride=1)
        self.bn21 = nn.BatchNorm2d(64)
        # conv22
        self.coords22 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv22 = nn.Conv2d(64,64,3,padding=1,stride=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.mp22 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                    
        # conv31
        self.coords31 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv31 = nn.Conv2d(64, 128, 3, padding=1,stride=1)
        self.bn31 = nn.BatchNorm2d(128)
        # conv32
        self.coords32 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1,stride=1)
        self.bn32 = nn.BatchNorm2d(128)
        self.ap32 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # out
        self.fc = nn.Linear(128, 10)  
    
    
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

        
        coords[:,:,0]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0),1.0)
        coords[:,:,1]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),0.0))),1.0),1.0)    
        
        coords[:,:,2]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0),1.0)
        coords[:,:,3]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),1.0))),1.0),0.0)
        
        coords[:,:,4]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0),1.0)
        coords[:,:,5]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),2.0))),1.0),-1.0)
        
        coords[:,:,6]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0),0.0)
        coords[:,:,7]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),3.0))),1.0),1.0)
        
        coords[:,:,10]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),1.0),0.0)
        coords[:,:,11]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),4.0))),1.0),-1.0)
        
        coords[:,:,12]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0),-1.0)
        coords[:,:,13]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),5.0))),1.0),1.0)
        
        coords[:,:,14]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),1.0),-1.0)
        coords[:,:,15]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),6.0))),1.0),0.0)
        
        coords[:,:,16]=torch.add(torch.mul(torch.cos(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),1.0),-1.0)
        coords[:,:,17]=torch.add(torch.mul(torch.sin(torch.add(theta,torch.mul(torch.div(PI[0],8.0),7.0))),1.0),-1.0)
        
        coords=coords.expand(batch_size,-1,-1,-1) 
        coords=coords.permute(0, 3, 1, 2)
        
        coords = coords.cuda()
        return coords
    
    
    def forward(self, x):       
        # conv11
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords11, weight=self.conv11.weight, padding=(1,1)) 
        x = self.bn11(x)
        x = F.relu(x)
    
        # conv12
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords12, weight=self.conv12.weight, padding=(1,1)) 
        x = self.bn12(x)
        x = F.relu(x)
        x = self.mp12(x)
    
        # conv21
        x=torchvision.ops.deform_conv2d(input=x, offset=self.coords21, weight=self.conv21.weight, padding=(1,1)) 
        x = self.bn21(x)
        x = F.relu(x)
        
        # conv22
        x=torchvision.ops.deform_conv2d(input=x, offset=self.coords22, weight=self.conv22.weight, padding=(1,1)) 
        x = self.bn22(x)
        x = F.relu(x)
        x = self.mp22(x)  
        
        # conv31
        x=torchvision.ops.deform_conv2d(input=x, offset=self.coords31, weight=self.conv31.weight, padding=(1,1)) 
        x = self.bn31(x)
        x = F.relu(x)

        # conv32
        x=torchvision.ops.deform_conv2d(input=x, offset=self.coords32, weight=self.conv32.weight, padding=(1,1)) 
        x = self.bn32(x)
        x = F.relu(x)
        x = self.ap32(x)
              
        x = self.fc(x.view(x.size()[:2]))
        
        return x   

def get_rot_inv_cnn(BATCH_SIZE):
    return Rot_Inv_ConvNet(BATCH_SIZE)

    