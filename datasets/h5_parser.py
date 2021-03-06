import h5py
import numpy as np
import torch

h5_name = "/media/liweijie/代码和数据/datasets/motionRetargeting/mocap_data.h5"

class h5Parser:
    def __init__(self,h5_name):
        self.f =  h5py.File(h5_name, "r") 

    def parse(self,group_name):
        # @return:q        np.ndarray with shape of T x J x 4
        # @return:p        np.ndarray with shape of T x J x 3
        # @return:glove_angle     np.ndarray with shape of T x 2 x 14
        ## Read needed data from mocap file

        l_wrist_pos = self.f[group_name + '/l_hd_pos'][:]
        l_wrist_quat = self.f[group_name + '/l_hd_quat'][:] # quaternion is (x,y,z,w), refer to quat_to_ndarray()
        l_elbow_pos = self.f[group_name + '/l_fr_pos'][:]
        l_elbow_quat = self.f[group_name + '/l_fr_quat'][:]
        l_shoulder_pos = self.f[group_name + '/l_up_pos'][:]
        l_shoulder_quat = self.f[group_name + '/l_up_quat'][:]

        r_wrist_pos = self.f[group_name + '/r_hd_pos'][:]
        r_wrist_quat = self.f[group_name + '/r_hd_quat'][:]
        r_elbow_pos = self.f[group_name + '/r_fr_pos'][:]  
        r_elbow_quat = self.f[group_name + '/l_fr_quat'][:]
        r_shoulder_pos = self.f[group_name + '/r_up_pos'][:]
        r_shoulder_quat = self.f[group_name + '/r_up_quat'][:]

        l_glove_angle = self.f[group_name + '/l_glove_angle'][:]
        r_glove_angle = self.f[group_name + '/r_glove_angle'][:]

        time = self.f[group_name + '/time'][:] # remember to store the timestamps information

        q =  np.stack([l_shoulder_quat,l_elbow_quat,l_wrist_quat,r_shoulder_quat,r_elbow_quat,r_wrist_quat],1)
        p =  np.stack([l_shoulder_pos,l_elbow_pos,l_wrist_pos,r_shoulder_pos,r_elbow_pos,r_wrist_pos],1)
        glove_angle = np.stack([l_glove_angle,r_glove_angle],1)

        # Transform the origin of postition
        origin = (l_shoulder_pos[0] + r_shoulder_pos[0]) / 2
        p = p - origin
        return q, p, glove_angle
    
    # def __del__(self):
    #     self.f.close()

# test
if __name__=="__main__":
    h5parser = h5Parser(h5_name)
    q, p, glove_angle = h5parser.parse("baozhu_1.bag")
    # print(f"q: {q}")
    print(f"p: {p}")