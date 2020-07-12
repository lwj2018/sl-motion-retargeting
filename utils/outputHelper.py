import torch
import numpy as np
from math import pi
from cfg import ROBOTHAND_LB,ROBOTHAND_UB
from cfg import WISEGLOVE_LB, WISEGLOVE_UB

# ROBOTHAND_LB = torch.Tensor(ROBOTHAND_LB)
# ROBOTHAND_UB = torch.Tensor(ROBOTHAND_UB)
# WISEGLOVE_LB = torch.Tensor(WISEGLOVE_LB)
# WISEGLOVE_UB = torch.Tensor(WISEGLOVE_UB)

def linear_map(x,min_h,max_h,min_r,max_r):
    return (x-min_h)/(max_h-min_h)*(max_r-min_r) + min_r

def  humanfinger2robotgoal(q_finger_human):
    """
        Direct mapping and linear scaling
        @param:q_finger_human            np.ndarray with shape of N x T x 2 x 14
        @param:q_finger_robot_goal    np.ndarray with shape of N x T x 2 x 12
    """
    q_finger_robot_goal = np.zeros(list(q_finger_human.shape[:-1])+[12]);
    q_finger_robot_goal[...,0] = linear_map(q_finger_human[...,3], WISEGLOVE_LB[...,3], WISEGLOVE_UB[...,3], ROBOTHAND_LB[...,0], ROBOTHAND_UB[...,0]);
    q_finger_robot_goal[...,1] = linear_map(q_finger_human[...,4], WISEGLOVE_LB[...,4], WISEGLOVE_UB[...,4], ROBOTHAND_LB[...,1], ROBOTHAND_UB[...,1]);
    q_finger_robot_goal[...,2] = linear_map(q_finger_human[...,6], WISEGLOVE_LB[...,6], WISEGLOVE_UB[...,6], ROBOTHAND_LB[...,2], ROBOTHAND_UB[...,2]);
    q_finger_robot_goal[...,3] = linear_map(q_finger_human[...,7], WISEGLOVE_LB[...,7], WISEGLOVE_UB[...,7], ROBOTHAND_LB[...,3], ROBOTHAND_UB[...,3]);
    q_finger_robot_goal[...,4] = linear_map(q_finger_human[...,9], WISEGLOVE_LB[...,9], WISEGLOVE_UB[...,9], ROBOTHAND_LB[...,4], ROBOTHAND_UB[...,4]);
    q_finger_robot_goal[...,5] = linear_map(q_finger_human[...,10], WISEGLOVE_LB[...,10], WISEGLOVE_UB[...,10], ROBOTHAND_LB[...,5], ROBOTHAND_UB[...,5]);
    q_finger_robot_goal[...,6] = linear_map(q_finger_human[...,12], WISEGLOVE_LB[...,12], WISEGLOVE_UB[...,12], ROBOTHAND_LB[...,6], ROBOTHAND_UB[...,6]);
    q_finger_robot_goal[...,7] = linear_map(q_finger_human[...,13], WISEGLOVE_LB[...,13], WISEGLOVE_UB[...,13], ROBOTHAND_LB[...,7], ROBOTHAND_UB[...,7]);
    q_finger_robot_goal[...,8] = (ROBOTHAND_LB[...,8] + ROBOTHAND_UB[...,8]) / 2.0;
    q_finger_robot_goal[...,9] = linear_map(q_finger_human[...,2], WISEGLOVE_LB[...,2], WISEGLOVE_UB[...,2], ROBOTHAND_LB[...,9], ROBOTHAND_UB[...,9]);
    q_finger_robot_goal[...,10] = linear_map(q_finger_human[...,0], WISEGLOVE_LB[...,0], WISEGLOVE_UB[...,0], ROBOTHAND_LB[...,10], ROBOTHAND_UB[...,10]);
    q_finger_robot_goal[...,11] = linear_map(q_finger_human[...,1], WISEGLOVE_LB[...,1], WISEGLOVE_UB[...,1], ROBOTHAND_LB[...,11], ROBOTHAND_UB[...,11]); 
    q_finger_robot_goal = q_finger_robot_goal * pi /180.0
    return q_finger_robot_goal
