import sys
sys.path.append("../utils")
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.h5_parser import h5Parser
from Quaternions import Quaternions
import matplotlib.pyplot as plt

h5_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/mocap_data.h5"
train_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/train.txt"
test_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/test.txt"

def linear_interpolate_q(quaternions,tgt_len):
    #@param:quaternions         Quaternions
    #＠return:result                       np.ndarray
    src_len = quaternions.shape[0]
    src_points = np.linspace(0,1,src_len)
    tgt_points = np.linspace(0,1,tgt_len)
    result = np.zeros([tgt_len]+list(quaternions.shape[1:])+[4])
    for i,tgt_point in enumerate(tgt_points):
        down_val = np.max(src_points[src_points<=tgt_point])
        up_val = np.min(src_points[src_points>=tgt_point]) 
        p = (tgt_point - down_val)/float(up_val-down_val) if up_val>down_val else 0.0
        quat1 = quaternions[src_points==down_val,...]
        quat2 = quaternions[src_points==up_val,...]
        result[i] = np.array(Quaternions.slerp(quat1,quat2,p))
    return result

def lerp(pos1,pos2,p):
    return p*pos2+(1-p)*pos1

def linear_interpolate(positions,tgt_len):
    #@param:positions        np.ndarray
    #@return:result                np.ndarray
    src_len = positions.shape[0]
    src_points = np.linspace(0,1,src_len)
    tgt_points = np.linspace(0,1,tgt_len)
    result = np.zeros([tgt_len]+list(positions.shape[1:]))
    for i,tgt_point in enumerate(tgt_points):
        down_val = np.max(src_points[src_points<=tgt_point])
        up_val = np.min(src_points[src_points>=tgt_point]) 
        p = (tgt_point - down_val)/float(up_val-down_val) if up_val>down_val else 0.0
        pos1 = positions[src_points==down_val,...]
        pos2 = positions[src_points==up_val,...]
        result[i] = np.array(lerp(pos1,pos2,p))
    return result

class MocapDataset(Dataset):

    def __init__(self,h5_name,list_file,length=28):
        self.h5_name = h5_name
        self.list_file = list_file
        self.length = length
        self.h5parser = h5Parser(h5_name)
        self.parse_list()
        
    def parse_list(self):
        list_file = open(self.list_file,"r")
        group_name_list = [record.rstrip('\n') for record in list_file.readlines()]
        self.group_name_list = group_name_list

    def __len__(self):
        return len(self.group_name_list)

    def __getitem__(self,index):
        # Parse from H5 file
        group_name = self.group_name_list[index]
        q, p, glove_angle = self.h5parser.parse(group_name)
        q = Quaternions(q)
        # Linear interpolate
        q = linear_interpolate_q(q, self.length)
        p = linear_interpolate(p, self.length) 
        glove_angle = linear_interpolate(glove_angle, self.length)
        # convert to torch.Tensor
        q = torch.Tensor(q)
        p = torch.Tensor(p)
        glove_angle = torch.Tensor(glove_angle)
        return q, p, glove_angle, group_name

# test 
if __name__=="__main__":
    train_set = MocapDataset(h5_name,train_list)
    train_set[0]


 