import torch
import math
def Rxa(a):
    rot = torch.eye(4,4).cuda()
    rot[0,0] = 1
    rot[1,1] = math.cos(a)
    rot[1,2] = -math.sin(a)
    rot[2,1] = math.sin(a)
    rot[2,2] = math.cos(a)
    return rot

def Ryb(b):
    rot = torch.eye(4,4).cuda()
    rot[0,0] = math.cos(b)
    rot[0,2] = math.sin(b)
    rot[1,1] = 1
    rot[2,0] = -math.sin(b)
    rot[2,2] = math.cos(b)
    return rot

def Rzc(c):
    rot = torch.eye(4,4).cuda()
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
    T = torch.eye(4,4).cuda()
    T[0,3] = x
    T[1,3] = y
    T[2,3] = z
    return T

def create_rot_with_axis_angle(angle,axis='z'):
    one = torch.ones(angle.size()).cuda()
    zero = torch.zeros(angle.size()).cuda()
    cos = torch.cos(angle).cuda()
    sin = torch.sin(angle).cuda()
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

def extract_position(T):
    """
        @param:T      a Tensor with shape of (...x4x4)    
        @return:position      a Tensor with shape of (...x3)    
    """
    position = T[...,0:3,3]
    return position

def extract_quaternion(T):
    """
        @param:T      a Tensor with shape of (...x4x4)    
        @return:quaternion      a Tensor with shape of (...x4)    
    """
    t00 = T[...,0,0]; t01 = T[...,0,1]; t02 = T[...,0,2];
    t10 = T[...,1,0]; t11 = T[...,1,1]; t12 = T[...,1,2];
    t20 = T[...,2,0]; t21 = T[...,2,1]; t22 = T[...,2,2];
    w = 0.5*torch.sqrt(t00+t11+t22+1)
    x = 0.5*torch.sign(t21-t12)*torch.sqrt(t00-t11-t22+1)
    y = 0.5*torch.sign(t02-t20)*torch.sqrt(t11-t00-t22+1)
    z = 0.5*torch.sign(t10-t01)*torch.sqrt(t22-t11-t00+1)
    quaternion = torch.stack([w,x,y,z],-1)
    return quaternion