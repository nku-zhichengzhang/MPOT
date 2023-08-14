import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import numpy as np

import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from easydict import EasyDict
from torchvision import models
from .backbone import Decoder
from .aspp import ASPP

CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}
SIZE = (288*2,512*2)


def positionalencoding1d(d_model, length):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(100.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

def Soft_aggregation(ps, max_obj):
    '''
    ps: no x h x w
    max_obj: up boundary
    '''
    num_objects, H, W = ps.shape 
    em = torch.zeros(1, max_obj+1, H, W).to(ps.device)
    em[0, 0, :, :] =  torch.prod(1-ps, dim=0) # bg prob
    em[0,1:num_objects+1, :, :] = ps # obj prob
    em = torch.clamp(em, 1e-7, 1-1e-7) # truncate
    
    logit = torch.log((em /(1-em))) # ln (x / 1 - x)

    return logit



class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self, arch):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_corner = nn.Conv2d(4, 64*4, kernel_size=7, stride=2, padding=3, bias=False, groups=4)
        self.conv1_1_corner = nn.Conv2d(64*4, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.corner_pe = positionalencoding1d(64,4).view(1,64*4,1,1).cuda()
        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_bg, in_corner):
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        bg = torch.unsqueeze(in_bg, dim=1).float()
        
        f = F.interpolate(in_f, size=SIZE, mode='bilinear', align_corners=False)
        m = F.interpolate(m, size=SIZE, mode='bilinear', align_corners=False)
        bg = F.interpolate(bg, size=SIZE, mode='bilinear', align_corners=False)
        c = F.interpolate(in_corner, size=SIZE, mode='bilinear', align_corners=False)

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg) + self.conv1_1_corner(self.conv1_corner(c)+self.corner_pe)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024

        return r4, r3, r2, c1
 
class Encoder_Q(nn.Module):
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        # f = (in_f - self.mean) / self.std
        f = F.interpolate(in_f, size=SIZE, mode='bilinear', align_corners=False)

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/16, 1024

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class PriorSE(nn.Module):
    def __init__(self, planes):
        super(PriorSE, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x, plane):
        residual = x
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = x * F.interpolate(plane,size=x.shape[2:],mode='bilinear',align_corners=False)
        out = residual + x
        return out

class Decoder(nn.Module):
    def __init__(self, inplane, mdim, expand):
        super(Decoder, self).__init__()
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128 * expand, mdim)
        self.RF2 = Refine(64 * expand, mdim)
        
        self.PRF3 = Refine(128 * expand, mdim)
        self.PRF2 = Refine(64 * expand, mdim)

        self.rASPP = ASPP(4*mdim, nn.GroupNorm)


        cls_tower, cor_tower = [], []
        for _ in range(4):
            cls_tower.append(
                nn.Conv2d(mdim, mdim, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, mdim))
            cls_tower.append(nn.ReLU())
            
            cor_tower.append(
                nn.Conv2d(mdim, mdim, 3, padding=1, bias=False)
            )
            cor_tower.append(nn.GroupNorm(32, mdim))
            cor_tower.append(nn.ReLU())

        
        self.cls_tower = nn.Sequential(*cls_tower)
        self.cor_tower = nn.Sequential(*cor_tower)
                
        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)
        self.corner2 = nn.Conv2d(mdim, 4, kernel_size=(3,3), padding=(1,1), stride=1)

        
        self._init_weight()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, r4, r3, r2, r1, f, is_train):
        m4 = self.ResMM(self.rASPP(r4))
        cls_m4 = self.cls_tower(F.relu(m4))# out: 1/16, 256
        cor_m4 = self.cor_tower(F.relu(m4))
        
        cls_m3 = self.RF3(r3, cls_m4) # out: 1/8, 256
        cls_m2 = self.RF2(r2, cls_m3) # out: 1/4, 256

        cor_m3 = self.PRF3(r3, cor_m4)
        cor_m2 = self.PRF2(r2, cor_m3)      
        
        logits2 = self.pred2(F.relu(cls_m2))
        ps2 = F.softmax(logits2, dim=1)
        p2 = F.interpolate(ps2[:,1:2], size=f.shape[2:], mode='bilinear', align_corners=False)

        corner2 =  self.corner2(F.relu(cor_m2))
        ucorner2 = F.sigmoid(F.interpolate(corner2, size=f.shape[2:], mode='bilinear', align_corners=False))

        return EasyDict({'p2':p2,'c2':ucorner2})

        
class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        
        
    def forward(self, m_in, m_out, q_in, q_out, train=False):  # m_in: o,c,t,h,w
        _, _, H, W = q_in.size()
        no, centers, C = m_in.size()
        _, _, vd = m_out.shape
 
        qi = q_in.view(-1, C, H*W) 
        p = torch.bmm(m_in, qi) # no x centers x hw
        p = p / math.sqrt(C)
        p = torch.softmax(p, dim=1) # no x centers x hw

        mo = m_out.permute(0, 2, 1) # no x c x centers 
        mem = torch.bmm(mo, p) # no x c x hw
        mem = mem.view(no, vd, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p
    
class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()

        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class MetaClassifier(nn.Module):

    def __init__(self, channels_in, channels_mem):
        super(MetaClassifier, self).__init__()
        self.cin = channels_in
        self.cm = channels_mem

        self.convP = nn.Conv2d(channels_in, channels_mem, kernel_size=1, padding=0, stride=1)
        self.convM = nn.Sequential(
            nn.Conv2d(channels_mem, channels_mem, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            ResBlock(indim=channels_mem),
            nn.Conv2d(channels_mem, channels_mem, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            ResBlock(indim=channels_mem),
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels_mem, 1)

    def forward(self, feat_ref, feat):

        feat_in = torch.cat([feat_ref, feat], dim=1)
        featP = F.relu(self.convP(feat_in))
        featM = self.convM(featP)
        output = torch.sigmoid(self.fc(self.pool(featM).squeeze()))

        return output

class STAN(nn.Module):
    def __init__(self, opt):
        super(STAN, self).__init__()

        keydim = opt.keydim
        valdim = opt.valdim
        arch = opt.arch

        expand = CHANNEL_EXPAND[arch]

        self.Encoder_M = Encoder_M(arch) 
        self.Encoder_Q = Encoder_Q(arch)

        self.keydim = keydim
        self.valdim = valdim

        self.KV_M_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)

        self.Memory = Memory()
        self.Decoder = Decoder(2*valdim, 256, expand)

        self.sigma = nn.Parameter(torch.zeros(3))
        
    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, corners, num_objects): 
        # memorize a frame 
        # maskb = prob[:, :num_objects, :, :]
        # make batch arg list
        frame_batch = []
        mask_batch = []
        bg_batch = []

        for o in range(1, num_objects+1): # 1 - no
            frame_batch.append(frame)
            mask_batch.append(masks[:,o])
        for o in range(1, num_objects+1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))

        # make Batch
        frame_batch = torch.cat(frame_batch, dim=0)
        mask_batch = torch.cat(mask_batch, dim=0)
        bg_batch = torch.cat(bg_batch, dim=0)

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch, corners) # no, c, h, w
        _, c, h, w = r4.size()
        memfeat = r4
        k4, v4 = self.KV_M_r4(memfeat)
        k4 = k4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.keydim)
        v4 = v4.permute(0, 2, 3, 1).contiguous().view(num_objects, -1, self.valdim)
        
        return k4, v4, r4

    def segment(self, frame, keys, values, num_objects, max_obj, is_train): 
        # segment one input frame

        r4, r3, r2, r1 = self.Encoder_Q(frame)
        n, c, h, w = r4.size()
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1)
        r3e, r2e, r1e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1), r1.expand(num_objects,-1,-1,-1)

        m4, _ = self.Memory(keys, values, k4e, v4e, is_train)

        output = self.Decoder(m4, r3e, r2e, r1e, frame, is_train)
        output['p2'] = Soft_aggregation(output['p2'][:, 0], max_obj)

        return output

    def forward(self, frame, mask=None, corner=None, keys=None, values=None, num_objects=None, max_obj=None, is_train=None):

        if mask is not None: # keys
            return self.memorize(frame, mask, corner, num_objects)
        else:
            return self.segment(frame, keys, values, num_objects, max_obj, is_train)

