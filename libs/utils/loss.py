import math
import torch
import torch.nn as nn
from torch.nn import functional as f

from .utility import mask_iou


def DiffLoss(self, input_a, input_b):
    assert (input_a.dim() == 4)  # num_frames, num_sequences, c, h, w
    assert (input_b.dim() == 4)
    norm_a = f.normalize(input_a, p=2, dim=[1,2,3])
    norm_b = f.normalize(input_b, p=2, dim=[1,2,3])

    return torch.mean(torch.sum(norm_a * norm_b, dim=[1,2,3]))


def binary_entropy_loss(pred, target, num_object, eps=0.001, ref=None):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)
    loss = torch.mean(ce)

    return loss

def cross_entropy_loss(pred, mask, num_object, bootstrap=0.4, ref=None):

    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape
    pred = torch.clamp(pred, 1e-7, 1-1e-7)
    pred = -1 * torch.log(pred)
    # predF = -1 * torch.log(1 - pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)
    ce = pred[:, :num_object+1] * mask[:, :num_object+1] 
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def binary_cross_entropy_loss(pred, mask, num_object, bootstrap=0.4, ref=None):
    
    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, _, H, W = mask.shape
    pred = torch.clamp(pred, 1e-7, 1-1e-7)
    pred = -1 * torch.log(pred)
    predF = -1 * torch.log(1 - pred)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)
    ce = pred[:, :num_object+1] * mask[:, :num_object+1] + predF[:, :num_object+1] * (1 - mask[:, :num_object+1])
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def cls_balanced_cross_entropy_loss(pred, mask, num_object, bootstrap=0.4, ref=None):
    
    # pred: [N x 1+C x H x W]
    # mask: [N x 1+C x H x W] smoothed label
    N, C, H, W = mask.shape
    pred = torch.clamp(pred, 1e-7, 1-1e-7)
    onehot = torch.argmax(mask,dim=1)
    num_neg = onehot.eq(0).sum()
    num_pos = onehot.gt(0).sum()
    num = num_pos + num_neg
    pred = -1 * torch.log(pred)

    # bootstrap
    num = int(H * W * bootstrap)
    ce =  pred[:num_object+1] * mask[:num_object+1] 
    ce[:,0:1] *= num_pos/num
    ce[:,1:5] *= num_neg/num

    if ref is not None:
        valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().squeeze(0).view(num_object,1,1,1)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def cornermap_mse_loss(pred, corner, num_object, bootstrap=0.4, ref=None,mseloss=None):

    N, C, H, W = corner.shape
    pred = torch.clamp(pred, 1e-7, 1-1e-7)

    # bootstrap
    num = int(H * W * bootstrap)
    mse =  mseloss(pred[:num_object+1],corner[:num_object+1]) 

    if ref is not None:
        valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().squeeze(0).view(num_object,1,1,1)
        mse *= valid

    loss = torch.sum(mse, dim=1).view(N, -1)
    # mloss, _ = torch.sort(loss, dim=-1, descending=True)
    # loss = torch.mean(mloss[:, :num])
    loss = torch.mean(loss)

    return loss


# def cornermap_mse_loss(pred, corner, num_object, bootstrap=0.4, ref=None,mseloss=None):
    
#     N, C, H, W = corner.shape
#     pred = torch.clamp(pred, 1e-7, 1-1e-7)
    
#     # roi area
#     # pred = pred.mul(corner>0)
    
#     # bootstrap
#     num = int(H * W * bootstrap)
#     mse =  mseloss(pred[:num_object+1],corner[:num_object+1]) 

#     if ref is not None:
#         valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
#         valid = valid.float().squeeze(0).view(num_object,1,1,1)
#         mse *= valid

#     loss = torch.sum(mse, dim=1).view(N, -1)
#     mloss, _ = torch.sort(loss, dim=-1, descending=True)
#     loss = torch.mean(mloss[:, :num])
#     # loss = torch.mean(loss)

#     return loss


# def point_loss(pred, coord, mask, num_object, bootstrap=0.4, ref=None,mseloss=None):
#     N, _, H, W, _ = pred.shape
#     coord = torch.flip(coord.view(N,4,1,1,2),[-1]).repeat(1,1,H,W,1).to(pred.device)

#     # bootstrap
#     mse = torch.mean(mseloss(pred[:num_object+1],coord[:num_object+1]),[-1])
#     # roi
#     # num = int(torch.sum(mask[0]) * bootstrap)
#     if ref is not None:
#         valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
#         valid = valid.float().squeeze(0).view(num_object,1,1,1)
#         mse *= valid
#     loss = torch.masked_select(mse, mask.ge(1e-5))
#     # mloss, _ = torch.sort(loss, dim=-1, descending=True)
#     loss = torch.mean(loss)

#     return loss

def point_loss(pred, coord, mask, num_object, bootstrap=0.4, ref=None,mseloss=None):
    N, _, H, W, _ = pred.shape
    coord = torch.flip(coord.view(N,4,1,1,2),[-1]).repeat(1,1,H,W,1).to(pred.device)

    # bootstrap
    mse = torch.mean(mseloss(pred[:num_object+1],coord[:num_object+1]),[-1])
    # reg = torch.mean(torch.abs(coord[:num_object+1]),[-1])
    # roi
    # num = int(torch.sum(mask[0]) * bootstrap)
    if ref is not None:
        valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().squeeze(0).view(num_object,1,1,1)
        mse *= valid
        # reg *= valid
    
    ploss = torch.masked_select(mse, mask.ge(1e-5))
    # rloss = torch.masked_select(reg, mask.ge(1e-5))
    # mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(ploss)
    # +torch.mean(rloss)

    return loss

def cls_balanced_mse_loss(pred, mask, num_object, bootstrap=0.4, ref=None,mseloss=None):
    
    # pred: [N x 1+C x H x W]
    # mask: [N x 1+C x H x W] smoothed label
    N, C, H, W = mask.shape
    pred = torch.clamp(pred, 1e-7, 1-1e-7)
    onehot = torch.argmax(mask,dim=1)
    num_neg = onehot.eq(0).sum()
    num_pos = onehot.gt(0).sum()
    num = num_pos + num_neg
    # pred = -1 * torch.log(pred)

    # bootstrap
    num = int(H * W * bootstrap)
    mse =  mseloss(pred[:num_object+1],mask[:num_object+1]) 
    # mse[:,0:1] *= num_pos/num
    # mse[:,1:5] *= num_neg/num

    if ref is not None:
        valid = (torch.sum(ref[:,1:num_object+1].view(ref.shape[0], num_object, -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().squeeze(0).view(num_object,1,1,1)
        mse *= valid

    loss = torch.sum(mse, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def weighted_binary_cross_entropy_loss(p, mask, num_object, bootstrap=0.4, ref=None):
    
    # pred: [N x K x H x W]
    # mask: [N x K x H x W] one-hot encoded
    N, no, H, W = mask.shape
    p = torch.clamp(p, 1e-5, 1-1e-5)
    pred = -1 * torch.log(p)
    predF = -1 * torch.log(1 - p)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)
    alpha = mask[:, :num_object+1].le(0.01).sum()/(N*min(no,num_object)*H*W)
    # bootstrap
    num = int(H * W * bootstrap)
    ce = alpha*pred[:, :num_object+1] * mask[:, :num_object+1] + (1-alpha)*predF[:, :num_object+1] *(1-mask[:, :num_object+1])
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        valid = valid.float().unsqueeze(2).unsqueeze(3)
        ce *= valid

    loss = torch.sum(ce, dim=1).view(N, -1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:, :num])

    return loss

def mask_iou_loss(pred, mask, num_object, ref=None):

    N, K, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)
    start = 0 if K == num_object else 1

    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)

    for i in range(N):
        obj_loss = (1.0 - mask_iou(pred[i, start:num_object+start], mask[i, start:num_object+start], averaged=False))
        if ref is not None:
            obj_loss = obj_loss[valid[i, start:]]

        loss += torch.mean(obj_loss)

    loss = loss / N
    return loss

def smooth_l1_loss(pred, target, gamma=0.075):
    if pred is None or target is None:
        return torch.Tensor([0.]).cuda()
    diff = torch.abs(pred-target)
    diff[diff>gamma] -= gamma / 2
    diff[diff<=gamma] *= diff[diff<=gamma] / (2 * gamma)

    return torch.mean(diff)

def ref_mse_loss(pred,gt,num_object,ref,mseloss):
    # centerness
    No, _, H, W = pred.size()
    loss = torch.zeros(1).to(pred.device)
    start = 0 if No == num_object else 1
    
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        
    obj_loss = []
    for i in range(num_object):
        # select ROI area
        # roi = gt_mask[0,start+i:start+i+1].ge(0.5)
        # roi_rate = H*W/roi.sum()
        # roi = roi.unsqueeze(1).repeat(1,8,1,1)
        # # calculate MSE loss in ROI area
        # roi_mse_loss = mse_loss(pred[start + 8*i:start + 8*(i+1)] * roi,gt[0,start + 8*i:start + 8*(i+1)] * roi)
        # obj_loss.append(roi_rate * roi_mse_loss)
        y_pred = pred[start +i]
        y_true = gt[start + i]
        # pos = y_pred.gt(0.5).sum()
        ori_mse_loss = mseloss(y_pred,y_true)
        obj_loss.append(ori_mse_loss)
        
    if ref is not None:
        valid_loss = torch.stack(obj_loss)[valid[0, start+1:]]
        
    if valid_loss.size(0)>0:
        loss += torch.mean(valid_loss)
    
    return loss

def ref_bayes_loss(pred,gt,num_object,ref):
    # centerness
    No, _, H, W = pred.size()
    loss = torch.zeros(1).to(pred.device)
    start = 0 if No == num_object else 1
    
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        
    obj_loss = []
    for i in range(num_object):
        y_pred = pred[start +i]
        y_true = gt[start + i]
        # pos = y_pred.gt(0.5).sum()
        ori_mse_loss = bayesian_loss(y_pred,y_true)
        obj_loss.append(ori_mse_loss)
        
    if ref is not None:
        valid_loss = torch.stack(obj_loss)[valid[0, start+1:]]
        
    if valid_loss.size(0)>0:
        loss += torch.mean(valid_loss)
    
    return loss

def bayesian_loss(pred_corner,gt_corner):
    gauss_p = (pred_corner*gt_corner).view(4,-1).sum(-1)
    gauss_n = (pred_corner*(1-gt_corner)).view(4,-1).sum(-1)
    target_p = gt_corner.max(-1).values.max(-1).values
    target_n = torch.zeros(4).to(gt_corner.device)

    # print(gauss_p,gauss_n,target_p,target_n)
    num_p = gt_corner.ge(0.5).sum()
    num_n = gt_corner.le(0.5).sum()
    
    pos = nn.L1Loss()(gauss_p,target_p)
    neg = nn.L1Loss()(gauss_n,target_n)
    # print(num_p,num_n,pos,neg)
    loss = (num_n/(num_n+num_p))*pos + (num_p/(num_n+num_p))*neg
    return loss

def ref_cos_loss(pred,gt,num_object,ref):
    # centerness
    No, dim = pred.size()
    loss = torch.zeros(1).to(pred.device)
    start = 0 if No == num_object else 1
    
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        
    obj_loss = []
    for i in range(num_object):
        
        y_pred = pred[start+i:start+i+1]
        y_true = gt[start+i:start+i+1]
        ori_cos_loss = 0.5*(1-f.cosine_similarity(y_pred,y_true))
        obj_loss.append(ori_cos_loss)
        
    if ref is not None:
        valid_loss = torch.stack(obj_loss)[valid[0, start+1:]]
        
    if valid_loss.size(0)>0:
        loss += torch.mean(valid_loss)
    
    return loss

def ref_focal_mse_loss(pred,gt,gt_mask,num_object,ref,mse_loss):
    # centerness
    No, _, H, W = pred.size()
    loss = torch.zeros(1).to(pred.device)
    start = 0 if No == num_object else 1
    
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        
    obj_loss = []
    for i in range(num_object):
        # select ROI area
        # roi = gt_mask[0,start+i:start+i+1].ge(0.5)
        # roi_rate = H*W/roi.sum()
        # roi = roi.unsqueeze(1).repeat(1,8,1,1)
        # # calculate MSE loss in ROI area
        # roi_mse_loss = mse_loss(pred[start + 8*i:start + 8*(i+1)] * roi,gt[0,start + 8*i:start + 8*(i+1)] * roi)
        # obj_loss.append(roi_rate * roi_mse_loss)
        y_true = gt[0,start + 8*i:start + 8*(i+1)]
        y_pred = pred[start + 8*i:start + 8*(i+1)]
        pt = torch.where(torch.gt(y_true, 0.99), y_pred, 1 - y_pred)  
        ori_mse_loss = mse_loss(y_pred, y_true) * torch.pow(1. - pt, 0.5)
        obj_loss.append(ori_mse_loss)
        
    if ref is not None:
        valid_loss = torch.stack(obj_loss)[valid[0, start+1:]]
        
    if valid_loss.size(0)>0:
        loss += torch.mean(valid_loss)
    
    return loss

def ref_focal_loss(pred,gt,num_object,ref):
    # centerness
    No, _, H, W = pred.size()
    loss = torch.zeros(1).to(pred.device)
    start = 0 if No == num_object else 1
    
    if ref is not None:
        valid = (torch.sum(ref.view(ref.shape[0], ref.shape[1], -1), dim=-1) > 0).to(pred.device)
        
    obj_loss = []
    for i in range(num_object):
        obj_loss.append(focal_neg_loss(pred[start+i],gt[start+i]))
        
    if ref is not None:
        valid_loss = torch.stack(obj_loss)[valid[0, start+1:]]
        
    if valid_loss.size(0)>0:
        loss += torch.mean(valid_loss)
            
    return loss

def focal_neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    '''
    
    if pred is None or gt is None:
        return torch.Tensor([0.]).cuda()
    
    pred = pred.clamp(1e-7,1-1e-7)
    
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 2)

    loss = 0
    # print(pred.max())
    # print(torch.log(1-pred.max()))
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    # print(num_pos,pos_loss,neg_loss)

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def pot_criterion(pred, gt):
    '''
    pred (num_obj x 8)
    gt (num_obj x 8)
    '''
    pred = pred.view(-1,4,2)
    gt = gt.view(-1,4,2)
    
    res = torch.pow(pred-gt, 2)
    p_res = torch.pow(torch.sum(res,dim=-1),0.5)
    
    return p_res.mean(dim=1)
    
