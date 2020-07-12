import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
        Disctiminate the action is real or fake
        based on the trajectory of wrist and elbow (position + rotation)
    """
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv1d(28,64,3,2,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,2,padding=1)
        self.conv3 = nn.Conv1d(128,256,3,2,padding=1)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)     
        self.fc4 = nn.Linear(32,1 )

    def forward(self,x):
        # Shape of Input x is: N x T x 4 x 7
        x = x.flatten(start_dim=-2)
        # After permute, shape of x is: N x D(4x7) x T
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Linear layers
        x = x.permute(0,2,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.mean(dim=1)
        # After forward, shape of x is: N x 1
        return x