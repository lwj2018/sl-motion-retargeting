import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import *
from transformMatrixHelper import *

class Kinematics_for_yumi(nn.Module):

    def __init__(self,kinematic_tree):
        super(Kinematics_for_yumi,self).__init__()
        self.kinematic_tree = kinematic_tree
        self.translations = YUMI_TRANSLATIONS
        self.rotations = YUMI_ROTATIONS
        self.names = names
        self.Transforms = []
        for trasition, rotation in zip(self.translations,self.rotations):
            T = Txyz(trasition)
            R_static = Rrpy(rotation)
            T_static = torch.matmul(T,R_static)
            self.Transforms.append(T_static)

    def  forward(self,eulers:torch.Tensor):
        T = torch.eye(4,4)
        result = {}
        for i, (T_static, name) in enumerate(zip(self.Transforms,self.names)):
            angle = eulers[...,i]
            T_dynamic = create_rot_with_axis_angle(angle,'z')
            Transition = torch.matmul(T_static,T_dynamic)
            T = torch.matmul(T, Transition)
            result[name] = T
        return result

if __name__=="__main__":
    fk_yumi = Kinematics_for_yumi(YUMI_TRANSLATIONS,YUMI_ROTATIONS,YUMI_NAMES)
    with torch.no_grad():
        euler = torch.Tensor([0.6,1.2,0.1,1.5,0.2,1.2,2.3])
    eulers = euler.expand(2,5,7)
    eulers.requires_grad = True
    res = fk_yumi(eulers)
    Te = res[YUMI_ELBOW_R_NAME]
    print(Te)
    Tw = res[YUMI_WRIST_R_NAME]
    y = torch.randn(2,5,4,4)
    y.requires_grad = True
    loss = nn.MSELoss()
    l = loss(Te,y) + loss(Tw,y)
    l.backward()
