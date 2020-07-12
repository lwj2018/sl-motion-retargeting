import sys
sys.path.append("../utils")
# sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import *
from utils.transformMatrixHelper import *
# from models.cfg import *
# from models.transformMatrixHelper import *

# Origin is the median point between two shoulder joints 
ORIGIN = 0.5*(YUMI_TRANSLATIONS[0]+YUMI_TRANSLATIONS[7])

class Kinematics_for_yumi(nn.Module):

    def __init__(self,kinematic_tree):
        super(Kinematics_for_yumi,self).__init__()
        self.kinematic_tree = kinematic_tree
        self.translations = YUMI_TRANSLATIONS
        self.rotations = YUMI_ROTATIONS
        self.name2ind = YUMI_NAME2IND
        self.root = YUMI_ROOT
        self.origin = ORIGIN
        self.Transforms = []
        for trasition, rotation in zip(self.translations,self.rotations):
            T = Txyz(trasition)
            R_static = Rrpy(rotation)
            T_static = torch.matmul(T,R_static)
            self.Transforms.append(T_static)

    def dfs(self,parent,cur,result,eulers):
        """
            parent is string for link name
            cur is string for link name
            result is a dict maps name to Global Transform 
            eulers is a Tensor for joint angles
        """
        if parent == -1:
            result[cur] = Txyz(-self.origin)
        else:
            ind = self.name2ind[cur] - 1
            T_static = self.Transforms[ind]
            angle = eulers[...,ind]
            T_dynamic = create_rot_with_axis_angle(angle,'z')
            Transition = torch.matmul(T_static,T_dynamic)
            T = torch.matmul(result[parent],Transition)
            result[cur] = T
        for child in self.kinematic_tree[cur]:
            self.dfs(cur,child,result,eulers)

    def  forward(self,eulers:torch.Tensor):
        result = {}
        self.dfs(-1,self.root,result,eulers)
        T_left_elbow = result[YUMI_ELBOW_L_NAME]
        T_right_elbow = result[YUMI_ELBOW_R_NAME]
        T_left_wrist = result[YUMI_WRIST_L_NAME]
        T_right_wrist = result[YUMI_WRIST_R_NAME]
        T_elbow_wrist_r = torch.stack([T_left_elbow,T_left_wrist,T_right_elbow,T_right_wrist],2)
        Q_elbow_wrist_r = extract_quaternion(T_elbow_wrist_r)
        P_elbow_wrist_r = extract_position(T_elbow_wrist_r)
        return Q_elbow_wrist_r, P_elbow_wrist_r

if __name__=="__main__":
    fk_yumi = Kinematics_for_yumi(YUMI_KINEMATIC_TREE)
    with torch.no_grad():
        euler = torch.Tensor([0.6,1.5,0.1,1.5,0.2,1.2,2.3,0.6,1.5,0.1,1.5,0.2,1.2,2.3])
    eulers = euler.expand(2,5,14)
    eulers.requires_grad = True
    res = fk_yumi(eulers)
    Tsr = res[YUMI_SHOULDER_R_NAME]
    Ter = res[YUMI_ELBOW_R_NAME]
    Twr = res[YUMI_WRIST_R_NAME]
    Tsl = res[YUMI_SHOULDER_L_NAME]
    Tel = res[YUMI_ELBOW_L_NAME]
    Twl = res[YUMI_WRIST_L_NAME]
    Qel = extract_quaternion(Tel)
    Qwl = extract_quaternion(Twl)
    print(Twl)
    print(Twr)
    y = torch.randn(2,5,4,4)
    y.requires_grad = True
    loss = nn.MSELoss()
    l = loss(Ter,y) + loss(Twr,y)
    l.backward()
