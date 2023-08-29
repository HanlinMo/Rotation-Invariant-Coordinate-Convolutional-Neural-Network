from __future__ import absolute_import, division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

class DenseNet3(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(DenseNet3, self).__init__()
        
        ###################################################################################################
        #head
        self.coords_h1_1 = self.generate_coordinates(BATCH_SIZE, 64, 64)
        self.conv_h1_1 = nn.Conv2d(3, 24, kernel_size=(3,3), stride=2, padding=1, bias=False)
        self.maxpool_h1_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        ###################################################################################################
        #Block 1
        self.bn_b1_1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)
        self.coords_b1_1 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_1 = nn.Conv2d(24, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_2 = nn.BatchNorm2d(36)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_2 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_2 = nn.Conv2d(36, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_3 = nn.BatchNorm2d(48)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_3 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_3 = nn.Conv2d(48, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)

        self.bn_b1_4 = nn.BatchNorm2d(60)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_4 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_4 = nn.Conv2d(60, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_5 = nn.BatchNorm2d(72)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_5 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_5 = nn.Conv2d(72, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_6 = nn.BatchNorm2d(84)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_6 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_6 = nn.Conv2d(84, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_7 = nn.BatchNorm2d(96)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_7 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_7 = nn.Conv2d(96, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_8 = nn.BatchNorm2d(108)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_8 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_8 = nn.Conv2d(108, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_9 = nn.BatchNorm2d(120)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_9 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_9 = nn.Conv2d(120, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_10 = nn.BatchNorm2d(132)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_10 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_10 = nn.Conv2d(132, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_11 = nn.BatchNorm2d(144)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_11 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_11 = nn.Conv2d(144, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b1_12 = nn.BatchNorm2d(156)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b1_12 = self.generate_coordinates(BATCH_SIZE, 32, 32)
        self.conv_b1_12 = nn.Conv2d(156, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_t1_1 = nn.BatchNorm2d(168)
        #self.relu = nn.ReLU(inplace=True)
        self.conv_t1_1 = nn.Conv2d(168, 168, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool_t1_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        ###################################################################################################
        #Block 2
        self.bn_b2_1 = nn.BatchNorm2d(168)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_1 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_1 = nn.Conv2d(168, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_2 = nn.BatchNorm2d(180)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_2 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_2 = nn.Conv2d(180, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_3 = nn.BatchNorm2d(192)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_3 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_3 = nn.Conv2d(192, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)

        self.bn_b2_4 = nn.BatchNorm2d(204)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_4 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_4 = nn.Conv2d(204, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_5 = nn.BatchNorm2d(216)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_5 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_5 = nn.Conv2d(216, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_6 = nn.BatchNorm2d(228)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_6 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_6 = nn.Conv2d(228, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_7 = nn.BatchNorm2d(240)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_7 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_7 = nn.Conv2d(240, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_8 = nn.BatchNorm2d(252)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_8 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_8 = nn.Conv2d(252, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_9 = nn.BatchNorm2d(264)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_9 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_9 = nn.Conv2d(264, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_10 = nn.BatchNorm2d(276)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_10 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_10 = nn.Conv2d(276, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_11 = nn.BatchNorm2d(288)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_11 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_11 = nn.Conv2d(288, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b2_12 = nn.BatchNorm2d(300)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b2_12 = self.generate_coordinates(BATCH_SIZE, 16, 16)
        self.conv_b2_12 = nn.Conv2d(300, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
   
        self.bn_t2_1 = nn.BatchNorm2d(312)
        #self.relu = nn.ReLU(inplace=True)
        self.conv_t2_1 = nn.Conv2d(312, 312, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool_t2_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        ############################################################################################################
        #Block 3
        self.bn_b3_1 = nn.BatchNorm2d(312)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_1 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_1 = nn.Conv2d(312, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_2 = nn.BatchNorm2d(324)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_2 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_2 = nn.Conv2d(324, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_3 = nn.BatchNorm2d(336)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_3 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_3 = nn.Conv2d(336, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)

        self.bn_b3_4 = nn.BatchNorm2d(348)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_4 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_4 = nn.Conv2d(348, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_5 = nn.BatchNorm2d(360)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_5 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_5 = nn.Conv2d(360, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_6 = nn.BatchNorm2d(372)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_6 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_6 = nn.Conv2d(372, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_7 = nn.BatchNorm2d(384)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_7 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_7 = nn.Conv2d(384, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_8 = nn.BatchNorm2d(396)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_8 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_8 = nn.Conv2d(396, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_9 = nn.BatchNorm2d(408)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_9 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_9 = nn.Conv2d(408, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_10 = nn.BatchNorm2d(420)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_10 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_10 = nn.Conv2d(420, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_11 = nn.BatchNorm2d(432)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_11 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_11 = nn.Conv2d(432, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
        
        self.bn_b3_12 = nn.BatchNorm2d(444)
        #self.relu = nn.ReLU(inplace=True)
        self.coords_b3_12 = self.generate_coordinates(BATCH_SIZE, 8, 8)
        self.conv_b3_12 = nn.Conv2d(444, 12, kernel_size=(3,3), stride=1, padding=1, bias=False)
   
        
        ###############################################################################################
        #fc
        self.bn_fc_1 = nn.BatchNorm2d(456)
        self.avgpool_fc_1 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(456, 30)
        
        ###############################################################################################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    
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
        
        ###############################################################################
        #head
        out = torchvision.ops.deform_conv2d(input=x, offset=self.coords_h1_1, weight=self.conv_h1_1.weight, stride=2, padding=(1,1))
        out = self.maxpool_h1_1(out)
        
        ###############################################################################
        #Block1
        Temp = out
        out = self.relu(self.bn_b1_1(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_1, weight=self.conv_b1_1.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_2(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_2, weight=self.conv_b1_2.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_3(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_3, weight=self.conv_b1_3.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_4(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_4, weight=self.conv_b1_4.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_5(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_5, weight=self.conv_b1_5.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_6(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_6, weight=self.conv_b1_6.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_7(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_7, weight=self.conv_b1_7.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_8(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_8, weight=self.conv_b1_8.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_9(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_9, weight=self.conv_b1_9.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_10(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_10, weight=self.conv_b1_10.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_11(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_11, weight=self.conv_b1_11.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b1_12(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b1_12, weight=self.conv_b1_12.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        out = self.conv_t1_1(self.relu(self.bn_t1_1(out)))      
        out = self.avgpool_t1_1(out)
        
        #########################################################################################
        #Block 2
        Temp = out
        out = self.relu(self.bn_b2_1(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_1, weight=self.conv_b2_1.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_2(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_2, weight=self.conv_b2_2.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_3(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_3, weight=self.conv_b2_3.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_4(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_4, weight=self.conv_b2_4.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_5(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_5, weight=self.conv_b2_5.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_6(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_6, weight=self.conv_b2_6.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_7(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_7, weight=self.conv_b2_7.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_8(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_8, weight=self.conv_b2_8.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_9(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_9, weight=self.conv_b2_9.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_10(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_10, weight=self.conv_b2_10.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_11(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_11, weight=self.conv_b2_11.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b2_12(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b2_12, weight=self.conv_b2_12.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        out = self.conv_t2_1(self.relu(self.bn_t2_1(out)))
        out = self.avgpool_t2_1(out)
        
        #########################################################################################################
        #Block 3
        Temp = out
        out = self.relu(self.bn_b3_1(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_1, weight=self.conv_b3_1.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_2(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_2, weight=self.conv_b3_2.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_3(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_3, weight=self.conv_b3_3.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_4(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_4, weight=self.conv_b3_4.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_5(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_5, weight=self.conv_b3_5.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_6(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_6, weight=self.conv_b3_6.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_7(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_7, weight=self.conv_b3_7.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_8(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_8, weight=self.conv_b3_8.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_9(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_9, weight=self.conv_b3_9.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_10(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_10, weight=self.conv_b3_10.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_11(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_11, weight=self.conv_b3_11.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        Temp = out
        out = self.relu(self.bn_b3_12(out))
        out = torchvision.ops.deform_conv2d(input=out, offset=self.coords_b3_12, weight=self.conv_b3_12.weight, padding=(1,1))
        out = torch.cat([Temp, out], 1)
        
        ##################################################################################################################
        #fc
        out = self.relu(self.bn_fc_1(out))
        out = self.avgpool_fc_1(out)
        out = out.view(-1, 456)
        out = self.fc(out)
        
        return out
    