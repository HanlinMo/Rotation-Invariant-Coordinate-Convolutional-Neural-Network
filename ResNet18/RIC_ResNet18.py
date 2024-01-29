from __future__ import absolute_import, division
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import math

class RIC_ResNet(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(RIC_ResNet, self).__init__() 
        
        self.coords_b0_1 = self.generate_coordinates(BATCH_SIZE, 128, 128)
        self.conv_b0_1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn_b0_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        ####################################################################################
        self.coords_b1 = self.generate_coordinates(BATCH_SIZE, 64, 64)
        self.conv_b1_1 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn_b1_1 = nn.BatchNorm2d(64)
        self.conv_b1_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn_b1_2 = nn.BatchNorm2d(64)
        self.downsample_b1_1 = nn.Sequential()
        
        self.conv_b1_3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn_b1_3 = nn.BatchNorm2d(64)
        self.conv_b1_4 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.bn_b1_4 = nn.BatchNorm2d(64)
        self.downsample_b1_2 = nn.Sequential()
        
        #######################################################################################
        self.coords_b2 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn_b2_1 = nn.BatchNorm2d(128)
        self.conv_b2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b2_2 = nn.BatchNorm2d(128)
        self.downsample_b2_1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128))
        
        self.conv_b2_3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b2_3 = nn.BatchNorm2d(128)
        self.conv_b2_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b2_4 = nn.BatchNorm2d(128)
        self.downsample_b2_2 = nn.Sequential()
        
        #######################################################################################
        self.coords_b3 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn_b3_1 = nn.BatchNorm2d(256)
        self.conv_b3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b3_2 = nn.BatchNorm2d(256)
        self.downsample_b3_1 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256))
        
        self.conv_b3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b3_3 = nn.BatchNorm2d(256)
        self.conv_b3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b3_4 = nn.BatchNorm2d(256)
        self.downsample_b3_2 = nn.Sequential()
        
        #######################################################################################
        self.coords_b4 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn_b4_1 = nn.BatchNorm2d(512)
        self.conv_b4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b4_2 = nn.BatchNorm2d(512)
        self.downsample_b4_1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(512))
        
        self.conv_b4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b4_3 = nn.BatchNorm2d(512)
        self.conv_b4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn_b4_4 = nn.BatchNorm2d(512)
        self.downsample_b4_2 = nn.Sequential()
        
        #######################################################################################
        
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(512, 30)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)
                
    
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
        
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b0_1, weight=self.conv_b0_1.weight, padding=(1,1))
        x = self.bn_b0_1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        ####################################################################################        
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b1, weight=self.conv_b1_1.weight, padding=(1,1)) 
        x = self.bn_b1_1(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b1, weight=self.conv_b1_2.weight, padding=(1,1)) 
        x = self.bn_b1_2(x)
        x += self.downsample_b1_1(Temp)
        x = self.relu(x)
        
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b1, weight=self.conv_b1_3.weight, padding=(1,1)) 
        x = self.bn_b1_3(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b1, weight=self.conv_b1_4.weight, padding=(1,1)) 
        x = self.bn_b1_4(x)
        x += self.downsample_b1_2(Temp)
        x = self.relu(x)
        
        #################################################################################### 
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b2, weight=self.conv_b2_1.weight, stride=2, padding=(1,1)) 
        x = self.bn_b2_1(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b2, weight=self.conv_b2_2.weight, padding=(1,1)) 
        x = self.bn_b2_2(x)
        x += self.downsample_b2_1(Temp)
        x = self.relu(x)
        
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b2, weight=self.conv_b2_3.weight, padding=(1,1)) 
        x = self.bn_b2_3(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b2, weight=self.conv_b2_4.weight, padding=(1,1)) 
        x = self.bn_b2_4(x)
        x += self.downsample_b2_2(Temp)
        x = self.relu(x)
        
        #################################################################################### 
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b3, weight=self.conv_b3_1.weight, stride=2, padding=(1,1)) 
        x = self.bn_b3_1(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b3, weight=self.conv_b3_2.weight, padding=(1,1)) 
        x = self.bn_b3_2(x)
        x += self.downsample_b3_1(Temp)
        x = self.relu(x)
        
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b3, weight=self.conv_b3_3.weight, padding=(1,1)) 
        x = self.bn_b3_3(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b3, weight=self.conv_b3_4.weight, padding=(1,1)) 
        x = self.bn_b3_4(x)
        x += self.downsample_b3_2(Temp)
        x = self.relu(x)
        
        #################################################################################### 
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b4, weight=self.conv_b4_1.weight, stride=2, padding=(1,1)) 
        x = self.bn_b4_1(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b4, weight=self.conv_b4_2.weight, padding=(1,1)) 
        x = self.bn_b4_2(x)
        x += self.downsample_b4_1(Temp)
        x = self.relu(x)
        
        Temp = x
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b4, weight=self.conv_b4_3.weight, padding=(1,1)) 
        x = self.bn_b4_3(x)
        x = self.relu(x)
        x = torchvision.ops.deform_conv2d(input=x, offset=self.coords_b4, weight=self.conv_b4_4.weight, padding=(1,1)) 
        x = self.bn_b4_4(x)
        x += self.downsample_b4_2(Temp)
        x = self.relu(x)
        
        #################################################################################### 
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
    
def RIC_ResNet18(BATCH_SIZE):
    return RIC_ResNet(BATCH_SIZE)