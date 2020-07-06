import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from h5_parser import h5Parser

h5_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/mocap_data.h5"
train_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/train.txt"
test_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/test.txt"


class MocapDataset(Dataset):

    def __init__(self,h5_name,list_file,length=128):
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
        group_name = self.group_name_list[index]
        q, p = self.h5parser.parse(group_name)
        q = torch.Tensor(q)
        p = torch.Tensor(p)
        return q, p

# test 
if __name__=="__main__":
    train_set = MocapDataset(h5_name,train_list)
    print(train_set[0])


 