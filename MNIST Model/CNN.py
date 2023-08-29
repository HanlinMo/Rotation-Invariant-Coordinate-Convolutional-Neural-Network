from __future__ import absolute_import, division
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        # conv12
        self.conv12 = nn.Conv2d(32,32, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.mp12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #conv21
        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        # conv22
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.mp22 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # conv31
        self.conv31 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        # conv32
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(128)
        self.ap32 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)       
        self.fc = nn.Linear(128, 9)

    def forward(self, x):
            
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
   
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.mp12(x)
        
        x = self.conv21(x)
        x = self.bn21(x)
        x = F.relu(x)

        x = self.conv22(x)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.mp22(x)
        
        x = self.conv31(x)
        x = self.bn31(x)
        x = F.relu(x)
    
        x = self.conv32(x)
        x = self.bn32(x)
        x = F.relu(x)       
        
        x = self.ap32(x)
        x = self.fc(x.view(x.size()[:2]))
            
        return x
        
def get_cnn():
    return ConvNet()