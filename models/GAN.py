import sys
sys.path.append("../utils")
# sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import opt
from cfg import YUMI_KINEMATIC_TREE
from utils.Recorder import Recorder
from utils.Averager import AverageMeter
from models.EncoderDecoder_human import Encoder_h,Decoder_h
from models.EncoderDecoder_robot import Encoder_r,Decoder_r
from models.Discriminate_model import Discriminator
from models.Kinematics import Kinematics_for_yumi
from models.loss import Generator_loss, Discriminator_loss
# from Recorder import Recorder
# from Averager import AverageMeter

class GAN_model(nn.Module):

    def __init__(self):
        super(GAN_model,self).__init__()
        self.Eh = Encoder_h()
        self.Dh = Decoder_h()
        self.Er = Encoder_r()
        self.Dr = Decoder_r()
        self.C = Discriminator()
        self.Fk = Kinematics_for_yumi(YUMI_KINEMATIC_TREE)
        self.G_loss = Generator_loss()
        self.D_loss = Discriminator_loss()
        self.optimizer_G = torch.optim.Adam(self.get_G_optim_policies(opt.lr))
        self.optimizer_D = torch.optim.Adam(self.C.parameters(),lr=opt.lr)

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self,Qh,Ph):
        """
            @param:Qh        torch.Tensor with shape of N x T x J x 4
            @param:Ph         torch.Tensor with shape of N x T x J x 3
        """
        Phi_h = self.Eh(Qh)
        Qh_hat = self.Dh(Phi_h)
        Jr = self.Dr(Phi_h)
        Qr_ew, Pr_ew = self.Fk(Jr)
        QPr_ew = torch.cat([Qr_ew,Pr_ew],-1)
        Qh_ew = Qh[:,:,[1,2,4,5],:]
        Ph_ew = Ph[:,:,[1,2,4,5],:]
        QPh_ew = torch.cat([Qh_ew,Ph_ew],-1)
        Real_pred = self.C(QPh_ew)
        Fake_pred = self.C(QPr_ew)
        Phi_r = self.Er(Jr)
        # print(f"Ph: {Ph_ew}")
        # print(f"Pr: {Pr_ew}")
        return Qh, Ph, Qh_hat, Qh_ew, Ph_ew,Jr, Qr_ew, Pr_ew, Phi_h, Phi_r, Real_pred, Fake_pred

    def backward_G(self,outputs):
        Ltotal, Lrec, Lltc, Lee, Ladv  = self.G_loss(*outputs)
        Ltotal.backward()
        return Ltotal, Lrec, Lltc, Lee, Ladv 

    def backward_D(self,outputs):
        Ladv = self.D_loss(*outputs)
        Ladv.backward()
        return Ladv

    def calculate_losses(self,outputs):
        Ltotal, Lrec, Lltc, Lee, Ladv_G  = self.G_loss(*outputs)
        Ladv_D = self.D_loss(*outputs)
        return Ltotal, Lrec, Lltc, Lee, Ladv_G, Ladv_D

    def optimize_parameters(self,Qh,Ph):
        # forward
        outputs = self(Qh,Ph)
        self.set_requires_grad([self.C], False)
        self.optimizer_G.zero_grad()
        Ltotal, Lrec, Lltc, Lee, Ladv_G  = self.backward_G(outputs)
        self.optimizer_G.step()

        outputs = self(Qh,Ph)
        self.set_requires_grad([self.C], True)
        self.optimizer_D.zero_grad()
        Ladv_D = self.backward_D(outputs)
        self.optimizer_D.step()
        return Ltotal, Lrec, Lltc, Lee, Ladv_G, Ladv_D

    def get_G_optim_policies(self,lr):
        return [
            {'params':self.Eh.parameters(),'lr':lr},
            {'params':self.Dh.parameters(),'lr':lr},
            {'params':self.Er.parameters(),'lr':lr},
            {'params':self.Dr.parameters(),'lr':lr}
        ]

def train_one_epoch(model, trainloader, device, epoch, log_interval, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_ltotal = AverageMeter()
    avg_lrec = AverageMeter()
    avg_lltc = AverageMeter()
    avg_lee = AverageMeter()
    avg_ladv_g = AverageMeter()
    avg_ladv_d = AverageMeter()
    # Set trainning mode
    model.train()
    # Create recorder
    averagers = [avg_ltotal,avg_lrec,avg_lltc,avg_lee,avg_ladv_g,avg_ladv_d]
    names = ['train Ltotal','train Lrec','train Lltc','train Lee','train Ladv G','train Ladv D']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)

    recoder.tik()
    recoder.data_tik()
    for i, data in enumerate(trainloader):
        # measure data loading time
        recoder.data_tok()

        # get the inputs
        Qh, Ph, glove_angles, group_names = data 
        Qh, Ph, glove_angles = [x.to(device) for x in (Qh, Ph, glove_angles)]

        # optimize parameters
        losses = model.optimize_parameters(Qh,Ph)
        Ltotal, Lrec, Lltc, Lee, Ladv_G, Ladv_D = losses

        # measure elapsed time
        recoder.tok()
        recoder.tik()
        recoder.data_tik()

        # update average value
        vals = [Ltotal.item(),Lrec.item(),Lltc.item(),Lee.item(),Ladv_G.item(),Ladv_D.item()]
        N = Qh.size(0)
        recoder.update(vals,count=N)

        if i==0 or i % log_interval == log_interval-1:
            recoder.log(epoch,i,len(trainloader))
            # Reset average meters 
            recoder.reset() 

def test_one_epoch(model, testloader, device, epoch, log_interval, writer, h5writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_ltotal = AverageMeter()
    avg_lrec = AverageMeter()
    avg_lltc = AverageMeter()
    avg_lee = AverageMeter()
    avg_ladv_g = AverageMeter()
    avg_ladv_d = AverageMeter()
    # Set eval mode
    model.eval()
    # Create recorder
    averagers = [avg_ltotal,avg_lrec,avg_lltc,avg_lee,avg_ladv_g,avg_ladv_d]
    names = ['test Ltotal','test Lrec','test Lltc','test Lee','test Ladv G','test Ladv D']
    recoder = Recorder(averagers,names,writer,batch_time,data_time)

    recoder.tik()
    recoder.data_tik()
    with torch.no_grad():
        for i, data in enumerate(testloader):
            # measure data loading time
            recoder.data_tok()

            # get the inputs
            Qh, Ph, glove_angles, group_names = data 
            Qh, Ph, glove_angles = [x.to(device) for x in (Qh, Ph, glove_angles)]

            # forward
            outputs = model(Qh,Ph)
            losses = model.calculate_losses(outputs)
            Qh, Ph, Qh_hat, Qh_ew, Ph_ew,Jr, Qr_ew, Pr_ew, Phi_h, Phi_r, Real_pred, Fake_pred = outputs
            Ltotal, Lrec, Lltc, Lee, Ladv_G, Ladv_D = losses

            # save file
            for group_name, joint_angle, glove_angle  in zip(group_names,Jr,glove_angles):
                h5writer.write(group_name, joint_angle, glove_angle)

            # measure elapsed time
            recoder.tok()
            recoder.tik()
            recoder.data_tik()

            # update average value
            vals = [Ltotal.item(),Lrec.item(),Lltc.item(),Lee.item(),Ladv_G.item(),Ladv_D.item()]
            N = Qh.size(0)
            recoder.update(vals,count=N)

            if i==0 or i % log_interval == log_interval-1:
                recoder.log(epoch,i,len(testloader),mode='Test')

    return avg_ltotal.avg

#test

if __name__=="__main__":
    model = GAN_model()
    Qh = torch.randn(2,10,6,4)
    Ph = torch.randn(2,10,6,3)
    outputs = model(Qh,Ph)
