import torch
import math
def Rxa(a):
    rot = torch.eye(4,4)
    rot[0,0] = 1
    rot[1,1] = math.cos(a)
    rot[1,2] = -math.sin(a)
    rot[2,1] = math.sin(a)
    rot[2,2] = math.cos(a)
    return rot

def Ryb(b):
    rot = torch.eye(4,4)
    rot[0,0] = math.cos(b)
    rot[0,2] = math.sin(b)
    rot[1,1] = 1
    rot[2,0] = -math.sin(b)
    rot[2,2] = math.cos(b)
    return rot

def Rzc(c):
    rot = torch.eye(4,4)
    rot[0,0] = math.cos(c)
    rot[0,1] = -math.sin(c)
    rot[1,0] = math.sin(c)
    rot[1,1] = math.cos(c)
    rot[2,2] = 1
    return rot

def Rrpy(rpy):
    c, b, a = rpy
    rot = torch.matmul(Rzc(a),Ryb(b))
    rot = torch.matmul(rot,Rxa(c))
    return rot

def Txyz(xyz):
    x,y,z = xyz
    T = torch.eye(4,4)
    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    return T

def create_rot_with_axis_angle(angle,axis='z'):
    one = torch.ones(angle.size())
    zero = torch.zeros(angle.size())
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    if axis == 'x':
        row0 = torch.stack([one,zero,zero,zero],-1)
        row1 = torch.stack([zero,cos,-sin,zero],-1)
        row2 = torch.stack([zero,sin,cos,zero],-1)
        row3 = torch.stack([zero,zero,zero,one],-1)
    elif axis == 'y':
        row0 = torch.stack([cos,zero,sin,zero],-1)
        row1 = torch.stack([zero,one,zero,zero],-1)
        row2 = torch.stack([-sin,zero,cos,zero],-1)
        row3 = torch.stack([zero,zero,zero,one],-1)
    elif axis == 'z':
        row0 = torch.stack([cos,-sin,zero,zero],-1)
        row1 = torch.stack([sin,cos,zero,zero],-1)
        row2 = torch.stack([zero,zero,one,zero],-1)
        row3 = torch.stack([zero,zero,zero,one],-1)
    rot = torch.stack([row0,row1,row2,row3],-2)
    return rot