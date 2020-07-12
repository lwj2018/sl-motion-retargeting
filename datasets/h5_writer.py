import h5py
import numpy as np
import torch
from utils.outputHelper import humanfinger2robotgoal

class h5Writer:
    def __init__(self,h5_name):
        self.f =  h5py.File(h5_name, "a") 

    def write(self,group_name,joint_angle,glove_angle):
        """
            @param:group_name             string
            @param:joint_angle                torch.Tensor with shape of T x 14(2x7)
            @param:glove_angle              torch.Tensor with shape of T x 2 x 14
        """
        joint_angle = joint_angle.detach().data.cpu().numpy()
        glove_angle = glove_angle.detach().data.cpu().numpy()
        joint_angle = joint_angle.reshape([joint_angle.shape[0]]+[2,-1])
        r_joint_angle = joint_angle[:,0,:]
        l_joint_angle = joint_angle[:,1,:]
        glove_angle = humanfinger2robotgoal(glove_angle)
        glove_angle = glove_angle.reshape([glove_angle.shape[0]]+[-1])
        angles = np.concatenate([l_joint_angle,r_joint_angle,glove_angle],-1)
        if group_name in self.f.keys():
            # the group already exists
            del self.f[group_name]
        group = self.f.create_group(group_name)
        group.create_dataset("arm_traj_1", data=angles, dtype=float)


