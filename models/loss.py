import torch
import torch.nn as nn
import torch.nn.functional as F
from cfg import *
from utils.transformMatrixHelper import *
from models.gan_loss import GANLoss

class Generator_loss(nn.Module):
    def __init__(self):
        super(Generator_loss,self).__init__()
        self.adv_loss = GANLoss(use_lsgan=True)

    def forward(self, Qh, Ph, Qh_hat, Qh_ew, Ph_ew,Jr, Qr_ew, Pr_ew, Phi_h, Phi_r, Real_pred, Fake_pred):
        Vr_ew = Pr_ew[:,1:,...] - Pr_ew[:,:-1,...]
        Vh_ew = Ph_ew[:,1:,...] - Ph_ew[:,:-1,...]

        Lrec = F.mse_loss(Qh,Qh_hat)
        Lltc = F.mse_loss(Phi_h,Phi_r)
        Lee = F.mse_loss(Ph_ew,Pr_ew) + F.mse_loss(Vh_ew,Vr_ew)
        Ladv = self.adv_loss(Fake_pred,target_is_real=True)
        # Ltotal = Lrec + Lltc + Lee + Ladv
        Ltotal = Ladv
        return Ltotal, Lrec, Lltc, Lee, Ladv

class Discriminator_loss(nn.Module):
    def __init__(self):
        super(Discriminator_loss,self).__init__()
        self.adv_loss = GANLoss(use_lsgan=True)

    def forward(self, Qh, Ph, Qh_hat, Qh_ew, Ph_ew,Jr, Qr_ew, Pr_ew, Phi_h, Phi_r, Real_pred, Fake_pred):
        Ladv = self.adv_loss(Real_pred,target_is_real=True) + self.adv_loss(Fake_pred,target_is_real=False)
        return Ladv

