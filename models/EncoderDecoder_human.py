import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Recorder import Recorder
from utils.Averager import AverageMeter
# from Recorder import Recorder
# from Averager import AverageMeter

class AE_h(nn.Module):
    def __init__(self):
        super(AE_h,self).__init__()
        self.e_h = Encoder_h()
        self.d_h = Decoder_h()

    def forward(self,x):
        x = self.e_h(x)
        x = self.d_h(x)
        return x

class Encoder_h(nn.Module):
    def __init__(self):
        super(Encoder_h,self).__init__()
        self.conv1 = nn.Conv1d(24,64,3,1,padding=1)
        self.conv2 = nn.Conv1d(64,128,3,2,padding=1)
        self.conv3 = nn.Conv1d(128,256,3,2,padding=1)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,64)     

    def forward(self,x):
        # Shape of input x is: N x T x J x 4
        x = x.flatten(start_dim=-2)
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

class Decoder_h(nn.Module):
    def __init__(self):
        super(Decoder_h,self).__init__()
        self.deconv1 = nn.ConvTranspose1d(64,128,3,2,padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(128,128,3,2,padding=1,output_padding=1)
        self.deconv3 = nn.ConvTranspose1d(128,256,3,1,padding=1)
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,24)     


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
        # After forward, shape of x is: N x T x J x 4
        x = x.view(x.size()[:2]+(-1,4))
        return x

def train_one_epoch(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    # Set trainning mode
    model.train()
    # Create recorder
    averagers = [avg_loss]
    names = ['train loss']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)

    recoder.tik()
    recoder.data_tik()
    for i, data in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs
        q, p = [x.to(device) for x in data]

        optimizer.zero_grad()
        # forward
        outputs = model(q)

        # compute the loss
        loss = criterion(outputs,q)
        # backward & optimize
        loss.backward()

        optimizer.step()

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [loss.item()]
        N = q.size(0)
        recoder.update(vals,count=N)

        if i==0 or i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset() 

def test_one_epoch(model, criterion, testloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    # Set eval mode
    model.eval()
    # Create recorder
    averagers = [avg_loss]
    names = ['test loss']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)

    recoder.tik()
    recoder.data_tik()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            recoder.data_tok()

            # get the inputs
            q, p = [x.to(device) for x in data]

            # forward
            outputs = model(q)

            # compute the loss
            loss = criterion(outputs,q)

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

            # update average value
            vals = [loss.item()]
            N = q.size(0)
            recoder.update(vals,count=N)

            if i==0 or i % log_interval == log_interval-1:
                recoder.log(epoch,i,len(testloader),mode='Test')

    return avg_loss.avg

if __name__=="__main__":
    e_h = Encoder_h()
    d_h = Decoder_h()
    x = torch.randn(4,128,6,4)
    x = e_h(x)
    x = d_h(x)