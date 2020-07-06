import h5py
import os
import os.path as osp

h5_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/mocap_data.h5"
train_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/train.txt"
test_list = "/media/liweijie/代码和数据/datasets/motionRetargeting/test.txt"
train_list = open(train_list,'w')
test_list = open(test_list,'w')

f = h5py.File(h5_name, "r") 
action_set = {}
action_count = 0
for record in f.keys():
    action = record.split('_')[0]
    if action == 'calib': continue
    if action not in action_set.keys():
        action_set[action] = 1
        action_count += 1
    if action_count%4==0:
        test_list.write(record+'\n')
    else:
        train_list.write(record+'\n')
print(len(action_set))


