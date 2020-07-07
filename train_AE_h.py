import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import AE_h
from models.EncoderDecoder_human import train_one_epoch
from models.EncoderDecoder_human import test_one_epoch
from datasets import MocapDataset
from torch.utils.tensorboard import SummaryWriter
from utils.ioUtils import *

# Path settings
# h5_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/mocap_data.h5"
# train_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/train.txt"
# test_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/test.txt"
h5_name = "/home/liweijie/Data/motion-retargeting/mocap_data.h5"
train_list = "/home/liweijie/Data/motion-retargeting/train.txt"
test_list = "/home/liweijie/Data/motion-retargeting/test.txt"
model_path = "./checkpoint"
# Hyper params
learning_rate = 1e-4
batch_size = 8
epochs = 1000
length = 128
# Options
store_name = 'AE_h'
checkpoint = None
log_interval = 5
device_list = '0'
num_workers = 0

# Use specific gpus
os.environ["CUDA_VISIBLE_DEVICES"]=device_list
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Cuda available: {}'.format(torch.cuda.is_available()))

# Use writer to record
writer = SummaryWriter(os.path.join('runs/'+store_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

min_loss = 0.0
start_epoch = 0

# Load data
trainset = MocapDataset(h5_name,train_list,length=length)
devset = MocapDataset(h5_name,test_list,length=length)
print("Dataset samples: {}".format(len(trainset)+len(devset)))
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
testloader = DataLoader(devset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
# Create model
model = AE_h().to(device)
if checkpoint is not None:
    start_epoch, min_loss = resume_model(model,checkpoint)
# Run the model parallelly
if torch.cuda.device_count() > 1:
    print("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
# Create loss criterion & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Start training
print("Training Started".center(60, '#'))
for epoch in range(start_epoch, epochs):
    # Train the model
    train_one_epoch(model, criterion, optimizer, trainloader, device, epoch, log_interval, writer)
    # Test the model
    loss = test_one_epoch(model, criterion, testloader, device, epoch, log_interval, writer)
    # Save model
    # remember min loss and save checkpoint
    is_best = loss>min_loss
    min_loss = max(loss, min_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best': min_loss
    }, is_best, model_path, store_name)
    print("Epoch {} Model Saved".format(epoch+1).center(60, '#'))

print("Training Finished".center(60, '#'))


