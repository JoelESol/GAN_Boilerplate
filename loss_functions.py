import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

def loss_weights():
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['gan'], alpha['dis'], alpha['style'], alpha['identity']= 1, 1, 0.75, 3

    return alpha

class Loss_Functions:
    def __init__(self):
        self.alpha = loss_weights()

    def swd_loss(self, perceptual, perceptual_converted, projection_dimension=64):
        swd_loss=0
        style_loss=0
        for p1, p2 in zip(perceptual[:-1], perceptual_converted[:-1]):
            s = p1.shape
            if s[0] > 1:
                proj = torch.randn(s[2], projection_dimension, device="cuda:0")
                proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
                p1 = torch.matmul(p1, proj)
                p2 = torch.matmul(p2, proj)
            p1 = torch.topk(p1, s[0], dim=0)[0]
            p2 = torch.topk(p2, s[0], dim=0)[0]
            dist = p1 - p2
            wdist = torch.mean(torch.mul(dist, dist))
            swd_loss += wdist
        style_loss +=self.alpha['style']*swd_loss
        return style_loss

    def content_loss(self, perceptual, perceptual_converted):
        content_perceptual_loss=0
        content_perceptual_loss += F.mse_loss(perceptual[-1], perceptual_converted[-1])
        return self.alpha['identity'] * content_perceptual_loss

    def identity_loss(self, imgs, converted_imgs):
        i_loss = 0
        i_loss += F.l1_loss(imgs, converted_imgs)
        return self.alpha['identity'] * i_loss

    def cycle(self, imgs, recon_imgs):
        cycle_loss = 0
        cycle_loss += F.l1_loss(imgs, recon_imgs)
        return self.alpha['cycle'] * cycle_loss
        
    def dis_patch(self, real, fake):
        dis_loss = 0
        #DCGAN loss
        dis_loss += self.alpha['dis'] * F.mse_loss(real, torch.ones_like(real))
        dis_loss += self.alpha['dis'] * F.mse_loss(fake, torch.zeros_like(fake))
        return dis_loss

    def dis(self, real, fake):
        dis_loss = 0
        dis_loss += self.alpha['dis'] * F.binary_cross_entropy(real, torch.ones_like(real))
        dis_loss += self.alpha['dis'] * F.binary_cross_entropy(fake, torch.zeros_like(fake))
        return dis_loss


    def gen(self, fake):
        gen_loss = 0
        gen_loss += self.alpha['gan'] * F.binary_cross_entropy(fake, torch.ones_like(fake))
        return gen_loss

    def task(self, pred, gt):
        task_loss = 0
        task_loss += F.cross_entropy(pred, gt, ignore_index=-1)
        return task_loss

