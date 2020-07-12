import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Recorder import Recorder
from utils.Averager import AverageMeter
# from Recorder import Recorder
# from Averager import AverageMeter

class Encoder_r(nn.Module):
    def __init__(self):
        super(Encoder_r,self).__init__()
        self.conv1 = nn.Conv1d(14,64,3,1,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,2,padding=1)
        self.conv3 = nn.Conv1d(128,256,3,2,padding=1)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,64)     

    def forward(self,x):
        # x = x.flatten(start_dim=-2)
        # After permute, shape of x is: N x D(Jx4) x T
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Linear layers
        x = x.permute(0,2,1)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.fc3(x)
        # After forward, shape of x is: N x T//4 x E(64)
        return x

class Decoder_r(nn.Module):
    def __init__(self):
        super(Decoder_r,self).__init__()
        self.deconv1 = nn.ConvTranspose1d(64,128,3,2,padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(128,128,3,2,padding=1,output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(128,256,3,1,padding=1)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,14)     


    def forward(self,x):
        # After permute, shape of x is: N x E(64) x T//4
        x = x.permute(0,2,1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # Linear layers
        x = x.permute(0,2,1)
        x = F.relu(self.fc1(x))
        x= F.relu(self.fc2(x))
        x = self.fc3(x)
        # After forward, shape of x is: N x T x 14(2x7)
        return x
