import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch
import os,sys
import os.path as osp
import shutil
import cv2
import random
import argparse
from collections import Counter
from threading import Thread


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from scipy.ndimage import gaussian_filter
from PIL import Image
from ..dataset.data import ROOT
from ..config import getCfg, sanity_check
from .logger import getLogger
# from .guass_map import G_map
from .guass_torch import make_mesh, make_gaussian, gaussian_radius
from tqdm import tqdm

logger = getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--cfg', default='config_1.yaml', type=str, help='path to config file')
    parser.add_argument('--local_rank', default=-1, type=int, help='process local rank, only used for distributed training')
    parser.add_argument('--seed', default=20, type=int, help='setting random seed')
    parser.add_argument('options', help='other configurable options', nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    opt = getCfg()
    opt.cfg = args.cfg
    if osp.exists(args.cfg):
        opt.merge_from_file(args.cfg)

    if len(args.options) > 0:
        assert len(args.options) % 2 == 0, 'configurable options must be key-val pairs'
        opt.merge_from_list(args.options)

    sanity_check(opt)
    setup_seed(args.seed)
    
    return opt, args.local_rank

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

def center_offset(center_map,corner_map,H,W,device,ref_mask,K=10):

    center_index = center_map > 0.1
    N = center_index.sum()
    if N==0:
        return None
    center_point = torch.flip(torch.nonzero(center_index),dims=[1]).unsqueeze(1).repeat(1,4,1)
    offset = torch.masked_select(corner_map, center_index.unsqueeze(0).repeat(8,1,1)).view(8,-1).transpose(0,1).view(-1,4,2)
    scores = torch.masked_select(center_map, center_index)
    offset[:,:,0] *= W
    offset[:,:,1] *= H
    coord = center_point - offset # -1 x 4 x (x,y)
    # scores = torch.index_select(center_map,dim=0,index=center_index)
    # scores = torch.zeros(N).cuda()
    # for i in range(N):
    #     cur_mask = np.zeros((H,W), dtype=np.uint8)
    #     co = coord[i].cpu().numpy()
    #     cv2.fillPoly(cur_mask, co.astype(np.int32).reshape(1,4,2), 1)
    #     scores[i] = b_mask_iou(cur_mask,ref_mask)
    # coord_score = torch.cat([coord,scores.unequeeze(-1)],dim=1)
    ind = scores.sort(descending=True).indices
    if ind.size(0) > K:
        coord_s = torch.index_select(coord, 0, ind[:K])
    else:
        coord_s = torch.index_select(coord, 0, ind)
    offset_coord = torch.mean(coord_s,dim=0)
    return offset_coord

def make_gaussian2(y_idx, x_idx, height, width, sigma=7):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1)
    x_idx = x_idx.transpose(0, 1)

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g

class arg2gauss(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return corner2gauss(input[0], input[1], input[2])
    
    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

def corner2gauss(corner_map, mesh, dmax):
    '''corner_map: no x 4 x h x w'''
    guass_map = torch.zeros_like(corner_map)
    no, _, H, W = guass_map.size()
    for o in range(no):
        x_idx,y_idx = torch.zeros(1,4).to(corner_map.device),torch.zeros(1,4).to(corner_map.device)
        for pt in range(4):
            cm = corner_map[o,pt]
            idx = torch.argmax(cm)
            x, y = idx % W, idx // W
            x_idx[0,pt]=x
            y_idx[0,pt]=y
            
        guass_map[o] = make_gaussian(y_idx,x_idx,mesh[0],mesh[1],dmax).view(4,H,W).to(corner_map.device)
    return guass_map
    
# def quad_nms(offset_coord, scores, nms_th=0.5):

# def deNormTensor(tensor):
#     mean = (0.485, 0.456, 0.406)
#     std = (0.207, 0.194, 0.199)
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
#     return tensor


def deNormTensor(tensor):
    mean = (0.485, 0.456, 0.406)
    std = (0.207, 0.194, 0.199)
    normed_tensor = torch.zeros_like(tensor)
    for id, (t, m, s) in enumerate(zip(tensor, mean, std)):
        normed_tensor[id] = (t*s)+m
            # The normalize code -> t.sub_(m).div_(s)
    return tensor
    
    
def calc_homography(ref_v4, cur_v4, ref_mask, cur_mask):
    batchSize, C, H, W = ref_v4.size()
    ref_mask = F.interpolate(ref_mask,size=(H,W))
    cur_mask = F.interpolate(cur_mask,size=(H,W))
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    X, Y = np.meshgrid(np.linspace(0,W-1,W),np.linspace(0,H-1,H))
    XYhom = np.stack([Y,X],axis=2)
    XYhom = torch.from_numpy(XYhom)
    
    homo_list=[]
    for o in range(1,ref_mask.size(1)):
        ref_o_mask = ref_mask[0,o].ge(0.9)
        cur_o_mask = cur_mask[0,o].ge(0.9)
        
        if cur_o_mask.sum()<500:
            return None
        ref_selected_v = torch.masked_select(ref_v4[0].permute(1,2,0),ref_o_mask.unsqueeze(-1).repeat(1,1,512)).view(-1,512)
        cur_selected_v = torch.masked_select(cur_v4[0].permute(1,2,0),cur_o_mask.unsqueeze(-1).repeat(1,1,512)).view(-1,512)
        ref_selected_v = F.normalize(ref_selected_v, p=2.0)
        cur_selected_v = F.normalize(cur_selected_v, p=2.0)
                
        ref_selected = torch.masked_select(XYhom,ref_o_mask.cpu().unsqueeze(-1).repeat(1,1,2)).view(-1,1,2).int()
        cur_selected = torch.masked_select(XYhom,cur_o_mask.cpu().unsqueeze(-1).repeat(1,1,2)).view(-1,1,2).int()

        match = torch.mm(ref_selected_v, cur_selected_v.T)
        src,dst = match.size()
        match_fla = match.view(-1)
        indxy = match_fla.topk(500).indices
        ind_src = (indxy / dst).int()
        ind_dst = indxy % dst
        
        

        src_pts = np.float32([ref_selected[ind_ref,0].cpu().numpy()[::-1] for ind_ref in ind_src]).reshape(-1, 1, 2)*2.5
        dst_pts = np.float32([cur_selected[ind_cur,0].cpu().numpy()[::-1] for ind_cur in ind_dst]).reshape(-1, 1, 2)*2.5
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        homo_list.append(M)
    return homo_list
    
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint'):
    
    filepath = os.path.join(checkpoint, filename + '.pth.tar')
    torch.save(state, filepath)
    logger.info('save model at {}'.format(filepath))


def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def build_pot_coord(frame,txt):
    
    with open(txt) as txt:
        lines = txt.readlines()

    inp = lines[frame].split()
    # print(inp)
    pts = np.zeros(8,dtype = np.float32)
    
    if (np.array(inp,dtype=float)==0).all() and frame+1 < len(lines):
        inp,inb = lines[frame-1].split(), lines[frame+1].split()
    else:
        inb = inp
    # print(inp,inb)
    for ii in range(4):
        pts[2*ii] = (float(inp[2*ii])+float(inb[2*ii]))/2
        pts[2*ii+1] = (float(inp[2*ii+1])+float(inb[2*ii+1]))/2
        
    return pts

def build_mpot_coord(frame,txt):
    # frame: img name, int
    lines = np.loadtxt(txt, dtype=str)
    obj = []
    pts = {}
    for line in lines:
        anno = line.split(',')
        if frame == int(anno[0]):
            if anno[11]=='':
                print(anno)
                print(txt)
            pts[int(anno[11])] = np.array(anno[2:10]).astype(float)
            obj.append(int(anno[11]))
    return pts, obj

def build_train_mpot_coord(frames,txt,max_obj):
    objs, ptss=[], []
    nf = len(frames)
    top_ptss = np.zeros((nf,max_obj,8),dtype = np.float32)
    f = np.zeros((nf,max_obj),dtype=bool)
    for idx, frame in enumerate(frames):
        pts_frame, objs_frame = build_mpot_coord(frame,txt)
        # pts_frame: int-np8 dict, objs: key list of dict
        ptss.append(pts_frame)
        if idx==0 or idx==len(frames)-1:
            objs.extend(objs_frame)
    if len(objs)==0:
        return None,None
    # ptss: list of int-np8 dict, len = num frame
    res = sorted(Counter(objs).items(), key=lambda x: x[1], reverse=True)
    sortres = [x for x in res if x[1]==2]
    random.shuffle(sortres)
    if len(sortres) < max_obj:
        return None,None
    obj_id = []
    # sorted obj_id-frequency list
    for i in range(min(len(sortres),max_obj)):

        top_id = sortres[i][0]
        obj_id.append(top_id)
        for j in range(nf):
            if top_id in ptss[j].keys():
                # gt
                # top_ptss[j,i] = ptss[j][top_id]
                # move pts to the inner, exclude the effect of corners
                center = np.mean(ptss[j][top_id].reshape(4,2),axis=0)
                pts_no_corner = 0.2*np.tile(center,(4,1)).reshape(8)+0.8*ptss[j][top_id]
                top_ptss[j,i] = pts_no_corner.astype(int)
                f[j,i]=True

    return top_ptss, f

def build_test_mpot_coord(frames,txt,init_txt,test_mode):
    lines = np.loadtxt(init_txt, dtype=str)[test_mode::2]
    no = lines.shape[0]
    obj_init = [int(l.split(',')[11]) for l in lines]
    obj_ord_dict = dict(zip(obj_init,range(no)))
    # obj_init: obj id in list     no:num objects     obj_ord_dict:obj_id-order
    
    objs, ptss=[], []
    nf = len(frames)
    top_ptss = np.zeros((nf,no,8),dtype = np.float32)
    f = np.zeros((nf,no),dtype=bool)
    for frame in frames:
        # print(frame)
        pts_frame, _ = build_mpot_coord(frame,txt)
        # pts_frame: int-np8 dict
        for pt in pts_frame.keys():
            top_ptss[frame-1,obj_ord_dict[pt]]=pts_frame[pt]
            f[frame-1,obj_ord_dict[pt]]=True

    return top_ptss, f, obj_init


def build_test_mpot_mask(init_txt,test_mode,size=(720,1280)):
    # testmode 0 first 1 biggest
    lines = np.loadtxt(init_txt, dtype=str)[test_mode::2]
    sampleframe = [int(l.split(',')[0])-1 for l in lines]
    firstframe = [int(l.split(',')[0])-1 for l in np.loadtxt(init_txt, dtype=str)[::2]]
    no = lines.shape[0]
    obj_init = [int(l.split(',')[11]) for l in lines]
    obj_ord_dict = dict(zip(obj_init,range(no)))
    masks = np.zeros((no,2)+size,dtype=np.uint8)
    for line in lines:
        # print(line)
        anno = line.split(',')
        name = obj_ord_dict[int(anno[11])]
        label = np.array(anno[2:10]).astype(float).astype(np.int32).reshape(1,4,2)
        cv2.fillPoly(masks[name,1],label,1)
        masks[name,0] = (masks[name,1]==0).astype(np.uint8)
    return masks, sampleframe, firstframe

def build_pot_mask(pts,size=(720, 1280)):
    
    mask = np.zeros(size, dtype=np.uint8)
    label = pts.astype(np.int32).reshape(1,4,2)
    cv2.fillPoly(mask, label, 1)
    
    return mask

def build_mpot_mask(coord, size=(720, 1280)):
    # coord: num frame x num obj x8
    nf, no, _ = coord.shape
    masks = np.zeros((nf, no+1,)+size, dtype=np.uint8)
    for i in range(nf):
        # fg
        for j in range(no):
            mask = np.zeros(size,dtype=np.uint8)
            label = coord[i,j].astype(np.int32).reshape(1,4,2)
            cv2.fillPoly(mask, label, 1)
            masks[i,j+1] = mask
        # bg
        masks[i,0] = (masks[i,1:].sum(0)==0).astype(np.uint8)
    
    return masks


def build_pot_corner(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)
    # corner = torch.zeros(4,size[1],size[0])

    # for i in range(4):
    y_idx = resize[0:1,1::2]
    x_idx = resize[0:1,::2]
    corner = make_gaussian(y_idx, x_idx, mesh[0].clone(), mesh[1].clone()).view(4,size[1],size[0])
    
    return corner

def build_pot_center(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)

    y_idx = resize[0:1,1::2].mean(1, keepdims=True)
    x_idx = resize[0:1,::2].mean(1, keepdims=True)
    center = make_gaussian(y_idx, x_idx, mesh[0].clone(), mesh[1].clone()).view(4,size[1],size[0])
    
    return center

def build_mpot_corner(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)

    y_idx = resize[0:1,1::2]
    x_idx = resize[0:1,::2]
    corner = make_gaussian(y_idx, x_idx, mesh[0].clone(), mesh[1].clone()).view(4,size[1],size[0])

    return corner

def build_mpot_center(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)

    y_idx = resize[0:1,1::2].mean(1, keepdims=True)
    x_idx = resize[0:1,::2].mean(1, keepdims=True)
    center = make_gaussian(y_idx, x_idx, mesh[0].clone(), mesh[1].clone()).view(1,size[1],size[0])

    return center

def build_mpot_corner_radius(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)

    y_idx = resize[0:1,1::2]
    x_idx = resize[0:1,::2]
    height = torch.max(y_idx)-torch.min(y_idx)
    width = torch.max(x_idx)-torch.min(x_idx)
    radius = gaussian_radius((height, width), 0.6)
    
    corner = make_gaussian(y_idx, x_idx, mesh[0].clone(), mesh[1].clone(), min(100,radius/3)).view(4,size[1],size[0])

    return corner

def build_mpot_center_radius(pts,size,mesh):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.view(1,8)

    y_idx = resize[0:1,1::2]
    my_idx = y_idx.mean(1, keepdims=True)
    x_idx = resize[0:1,::2]
    mx_idx = x_idx.mean(1, keepdims=True)
    height = torch.max(y_idx)-torch.min(y_idx)
    width = torch.max(x_idx)-torch.min(x_idx)
    radius = gaussian_radius((height, width), 0.6)
    
    center = make_gaussian(my_idx, mx_idx, mesh[0].clone(), mesh[1].clone(), min(100,radius/3)).view(1,size[1],size[0])

    return center



def build_pot_corner_center(pts,size,mesh,offset):
    '''
    pts: 8d array
    size: w,h
    '''
    resize = pts.reshape(1,8)
    corner = np.zeros((8,size[1],size[0]), dtype=np.float32)
    center = np.zeros((1,size[1],size[0]), dtype=np.float32)
    c = np.zeros(2)
    
    for i in range(4):
        pt = resize[0,2*i:2*i+2]
        c+=pt
        # corner[2*i:2*i+1] = gaussian_filter(offset[0:1]-pt[1],sigma=20)/size[1]
        # corner[2*i+1:2*i+2] = gaussian_filter(offset[1:2]-pt[0],sigma=20)/size[0]
        corner[2*i:2*i+1] = offset[0:1]-pt[0]/size[0]
        corner[2*i+1:2*i+2] = offset[1:2]-pt[1]/size[1]
    
    center[0] = G_map(size[0], size[1], int(0.5+c[0]/4), int(0.5+c[1]/4), mesh.copy())

    return corner, center

def Nrotate(angle,valuex,valuey,pointx,pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
    nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
    return nRotatex, nRotatey

def gen_offset(W,H):
    
    xx_ones = np.ones([1, 1, W], dtype=np.int32)
    yy_ones = np.ones([1, 1, H], dtype=np.int32)

    xx_range = np.arange(H, dtype=np.int32)
    yy_range = np.arange(W, dtype=np.int32)
    xx_range = xx_range[None, :, None]
    yy_range = yy_range[None, :, None]

    yy_channel = np.matmul(xx_range, xx_ones)/H
    xx_channel = np.matmul(yy_range, yy_ones)
    xx_channel = np.transpose(xx_channel,(0, 2, 1))/W
    
    xy_offset = np.vstack([xx_channel,yy_channel])
    return xy_offset

def pts_resize(pts,src_size,tar_size):
    
    w_rate = tar_size[0]/src_size[0]
    h_rate = tar_size[1]/src_size[1]
    
    for id,pt in enumerate(pts):
        for i in range(4):
            pts[id,2*i] = pt[2*i]*w_rate
            pts[id,2*i+1] = pt[2*i+1]*h_rate

    return pts

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def pts_corr(pts1,pts2,size,mesh):
    # pts1,pts2: 1x8
    # size: h1,w1,h2,w2
    correlation = np.zeros(size,dtype=float)

    pts1, pts2 = pts1.reshape(4,2).astype(np.float32), pts2.reshape(4,2).astype(np.float32)
    H = cv2.getPerspectiveTransform(pts1, pts2)
    
    src, des = np.zeros(size[0],size[1],dtype=int), np.zeros(size[2],size[3],dtype=int)
    cv2.fillPoly(src, [pts1], 1)
    cv2.fillPoly(des, [pts2], 1)
    
    for src_x in range(size[0]):
        for src_y in range(size[1]):
            if src[src_x,src_y]:
                src_pt = np.float32([[[src_x, src_y]]])
                des_pt = cv2.perspectiveTransform(src_pt, H).reshape(-1).astype(int)
                correlation[src_x,src_y] = G_map(size[3], size[2], des_pt[0], des_pt[1], mesh).copy()
    return correlation, src, des

def re_logits(logits, info):
    h, w = info['size']
    th, tw = logits.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    m = logits[0, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
    m = m.transpose((1, 2, 0))
    rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return rescale_mask

def out2mask(mask, info):
    
    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    m = mask[0, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
    m = m.transpose((1, 2, 0))
    rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)
    
    return fg

def write_mask(mask, info, opt, directory='results'):

    """
    mask: numpy.array of size [T x max_obj x H x W]
    """

    name = info['name']

    directory = os.path.join(opt.code_root, opt.checkpoint, directory)
    # print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.join(directory, opt.valset)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.makedirs(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])    

        if opt.save_indexed_format == 'index':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')

        elif opt.save_indexed_format == 'segmentation':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max()+1):
                seg[rescale_mask==k] = info['palette'][(k*3):(k+1)*3]

            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'heatmap':
            
            rescale_mask[rescale_mask<0] = 0.0
            rescale_mask = np.max(rescale_mask[:, :, 1:], axis=2)
            rescale_mask = (rescale_mask - rescale_mask.min()) / (rescale_mask.max() - rescale_mask.min()) * 255
            seg = rescale_mask.astype(np.uint8)
            # seg = cv2.GaussianBlur(seg, ksize=(5, 5), sigmaX=2.5)

            seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'mask':

            fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)

            seg = np.zeros((h, w, 3), dtype=np.uint8)
            seg[fg==1] = info['palette'][3:6][::-1]

            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        else:
            raise TypeError('unknown save format {}'.format(opt.save_indexed_format))
        
def write_mask_boxes(mask, box, info, opt, directory='results'):
    
    """
    mask: numpy.array of size [T x max_obj x H x W]
    box: numpy.array of size [T x max_obj x 8]
    """
    box = box.reshape(box.shape[0],box.shape[1],8)
    # print(box.shape,mask.shape)
    name = info['name']

    directory = os.path.join(opt.code_root, opt.checkpoint, directory)
    # print('saving to: '+directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.join(directory, opt.valset)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.makedirs(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor*h), int(factor*w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2
    

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # pts = pts_resize(box[t], (1,1), (w,h))
        pts = box[t]
        
        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])    

        if opt.save_indexed_format == 'index':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')

        elif opt.save_indexed_format == 'segmentation':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max()+1):
                seg[rescale_mask==k] = info['palette'][(k*3):(k+1)*3]

            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            
            # draw polylines
            for id,pt in enumerate(pts):
                # print(type(pt),pt.shape)
                
                p = pt.reshape((-1,1,2)).astype(np.int32)
                cv2.polylines(im,[p],True,(0,255,255))
                for i in range(4):
                    text = '#'+str(id)+'_'+str(i)
                    cv2.putText(im,text,tuple(p[i,0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=2)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'heatmap':
            
            rescale_mask[rescale_mask<0] = 0.0
            rescale_mask = np.max(rescale_mask[:, :, 1:], axis=2)
            rescale_mask = (rescale_mask - rescale_mask.min()) / (rescale_mask.max() - rescale_mask.min()) * 255
            seg = rescale_mask.astype(np.uint8)
            # seg = cv2.GaussianBlur(seg, ksize=(5, 5), sigmaX=2.5)

            seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'mask':

            fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)

            seg = np.zeros((h, w, 3), dtype=np.uint8)
            seg[fg==1] = info['palette'][3:6][::-1]

            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)

            im = cv2.addWeighted(resized, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        else:
            raise TypeError('unknown save format {}'.format(opt.save_indexed_format))

def write_mpot_mask_boxes(box, info, opt, directory='results'):
    
    """
    mask: numpy.array of size [T x max_obj x H x W]
    box: numpy.array of size [T x max_obj x 8]
    """
    box = box.reshape(box.shape[0],box.shape[1],8)
    # print(box.shape,mask.shape)
    name = info['name']

    directory = os.path.join(opt.code_root, opt.checkpoint, directory)
    # print('saving to: '+directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = os.path.join(directory, opt.valset)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.makedirs(video)

    h, w = info['size']

    

    for t in range(box.shape[0]):
        # m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        # m = m.transpose((1, 2, 0))
        # rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        # pts = pts_resize(box[t], (1,1), (w,h))
        pts = box[t]
        
        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])    
        
        if opt.save_indexed_format == 'segmentation':

            inp_img = cv2.imread(os.path.join(info['split'], name, 'seq1', output_name.replace('png', 'jpg')))
            resized = cv2.resize(inp_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # draw polylines
            for id,pt in enumerate(pts):
                
                p = pt.reshape((-1,1,2)).astype(np.int32)
                cv2.polylines(resized,[p],True,(0,255,255))
                for i in range(4):
                    text = '#'+str(id)+'_'+str(i)
                    cv2.putText(resized,text,tuple(p[i,0]),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=2)
            cv2.imwrite(os.path.join(video, output_name), resized)

class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def myarea(x1,x2,x3,x4):
    a = x1-x2
    d1 = math.hypot(a[0],a[1])  #用math.hypot()函数求距离     
    b =  x2-x3
    d2 = math.hypot(b[0],b[1])
    c = x3-x4
    d3 = math.hypot(c[0],c[1])
    d = x1-x4
    d4 = math.hypot(d[0],d[1])
    e = x1-x3
    d5 = math.hypot(e[0],e[1])
    # print(d1,d2,d3,d4,d5)        #输出各边的长度
    k1 = (d1+d4+d5)/2 
    k2 = (d2+d3+d5)/2 
    s1 = (k1*(k1-d1)*(k1-d4)*(k1-d5))**0.5 
    s2 = (k2*(k2-d2)*(k2-d3)*(k2-d5))**0.5 
    s = ((s1+s2)**0.5)/100
    return s

def isConvex(points) -> bool:
        size = len(points)
        pre = 0
        for i in range(size):
            x1 = points[(i + 1) % size][0] - points[i][0]
            x2 = points[(i + 2) % size][0] - points[i][0]
            y1 = points[(i + 1) % size][1] - points[i][1]
            y2 = points[(i + 2) % size][1] - points[i][1]
            cur = x1 * y2 - x2 * y1
            if cur != 0:
                if cur * pre < 0:
                    return False
                else:
                    pre = cur
        return True



def b_mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)       # I assume this is faster as mask1 == 1 is a bool array
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
    if mask1_area+mask2_area-intersection!=0:
        iou = intersection/(mask1_area+mask2_area-intersection)
        return iou
    else:
        return 0

def mask_iou(pred, target, averaged = True):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = inter / union

    if averaged:
        iou = torch.mean(iou)

    return iou

def adjust_learning_rate(optimizer, epoch, opt):

    if epoch in opt.milestone:
        opt.learning_rate *= 0.1
        for pm in optimizer.param_groups:
            pm['lr'] *= opt.learning_rate

def pointwise_dist(points1, points2):

    # compute the point-to-point distance matrix

    N, d = points1.shape
    M, _ = points2.shape

    p1_norm = torch.sum(points1**2, dim=1, keepdim=True).expand(N, M)
    p2_norm = torch.sum(points2**2, dim=1).unsqueeze(0).expand(N, M)
    cross = torch.matmul(points1, points2.permute(1, 0))

    dist = p1_norm - 2 * cross + p2_norm

    return dist

def furthest_point_sampling(points, npoints):

    """
    points: [N x d] torch.Tensor
    npoints: int

    """

    old = 0
    output_idx = []
    output = []
    dist = pointwise_dist(points, points)
    fdist, fidx = torch.sort(dist, dim=1, descending=True)

    for i in range(npoints):
        fp = 0
        while fp < points.shape[0] and fidx[old, fp] in output_idx:
            fp += 1

        old = fidx[old, fp]
        output_idx.append(old)
        output.append(points[old])

    return torch.stack(output, dim=0)

def split_mask_by_k(mask, k):

    """
    mask: [H x W] torch.Tensor (one-hot encoded or float)
    k: int

    ret: [k x H x W] torch.Tensor
    """

    if k == 0:
        return mask.unsqueeze(0)

    H, W = mask.shape
    # meshx = torch.Tensor([[i for i in range(W)]]).float().to(mask.device).expand(H, W)
    # meshy = torch.Tensor([[i] for i in range(H)]).float().to(mask.device).expand(H, W)
    meshx = torch.Tensor([[i for i in range(W)]]).float().cuda().expand(H, W)
    meshy = torch.Tensor([[i] for i in range(H)]).float().cuda().expand(H, W)
    mesh = torch.stack([meshx, meshy], dim=2)

    foreground = mesh[mask>0.5, :].view(-1, 2)

    # samples = furthest_point_sampling(foreground, k)

    npoints = foreground.shape[0]
    nidx = random.sample(range(npoints), k)
    samples = foreground[nidx, :]

    mesh = mesh.view(-1, 2)
    dist = pointwise_dist(mesh, samples)
    _, cidx = torch.min(dist, dim=1)
    cidx = cidx.view(H, W)

    output = []

    for i in range(k):
        output.append(((cidx == i) * (mask > 0.5)).float())

    return torch.stack(output, dim=0)

def mask_to_box(masks, num_objects):

    """
    convert a mask annotation to coarse box annotation

    masks: [N x (K+1) x H x W]
    """

    N, K, H, W = masks.shape
    output = masks.new_zeros(masks.shape)

    for n in range(N):
        for o in range(1+num_objects):
            for start_x in range(W):
                if torch.sum(masks[n, o, :, start_x]) > 0:
                    break

            for end_x in range(W-1, -1, -1):
                if torch.sum(masks[n, o, :, end_x]) > 0:
                    break

            for start_y in range(H):
                if torch.sum(masks[n, o, start_y, :]) > 0:
                    break

            for end_y in range(H-1, -1, -1):
                if torch.sum(masks[n, o, end_y, :]) > 0:
                    break

            if start_x <= end_x and start_y <= end_y:
                output[n, o, start_y:end_y+1, start_x:end_x+1] = 1

    return output