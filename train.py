from libs.dataset.data import (
        ROOT, 
        MAX_TRAINING_OBJ, 
        build_dataset, 
        multibatch_collate_fn, 
        test_multibatch_collate_fn, 
    )

from libs.dataset.transform_mpot import TrainTransform, TestTransform
from libs.utils.logger import AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, write_mpot_mask_boxes, save_checkpoint, adjust_learning_rate, deNormTensor

from libs.models.models import STAN
from apex import amp
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import make_grid
import libs.utils.logger as logger
import libs.dataset.dali as dali

import numpy as np
import os
import os.path as osp
import shutil
import time
import cv2
import copy

from tensorboardX import SummaryWriter
from progress.bar import Bar
from collections import OrderedDict
from skimage.measure import label, regionprops
from tqdm import tqdm

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
MAX_FLT = 1e6

opt, _ = parse_args()

coord_rate = 100

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# Use CUDA
use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'

print('Running code in: '+opt.gpu_id)
tag = 'resnet50_center_radius'
opt.output_dir = tag
LOG_DIR = os.path.join(opt.checkpoint,'log'+tag)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)    
os.makedirs(LOG_DIR)
writer = SummaryWriter(log_dir=LOG_DIR)


def main():

    # setup
    start_epoch = 0    
    train_iter = 0
    test_iter = 0
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
    use_multi = False

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    opt.output = osp.join(osp.join(opt.checkpoint, opt.output_dir))
    if not osp.exists(opt.output):
        os.mkdir(opt.output)

    logfile = osp.join(opt.checkpoint, opt.output_dir, opt.mode+'_log'+code_id+tag+'.txt')
    logger.setup(filename=logfile, resume=opt.resume != '')
    log = logger.getLogger(__name__)

    # Data
    log.info('Preparing dataset')
    
    log.info('coord_rate:',coord_rate)
    log.info(opt)
    log.info(tag)
    
    input_dim = tuple(opt.input_size)
    train_transformer = TrainTransform(size=input_dim)
    test_transformer = TestTransform(size=input_dim)

    datalist = []
    for dataset, freq, max_skip,dataset_objs in zip(opt.trainset, opt.datafreq, opt.max_skip, opt.objs):

        if opt.data_backend == 'DALI' and not dataset.startswith('DALI'):
            dataset = 'DALI' + dataset

        ds = build_dataset(
            name=dataset,
            train=True, 
            sampled_frames=opt.sampled_frames, 
            transform=train_transformer, 
            max_skip=max_skip, 
            samples_per_video=opt.samples_per_video,
            size=input_dim,
            max_obj=dataset_objs,
        )
        datalist += [copy.deepcopy(ds) for _ in range(freq)]

    trainset = data.ConcatDataset(datalist)

    testset = build_dataset(
        name=opt.valset,
        train=False, 
        split='val',
        tag=tag,
        transform=test_transformer, 
        samples_per_video=1,
        size=input_dim
        )

    if opt.data_backend == 'PIL':
        trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                      collate_fn=multibatch_collate_fn)
    elif opt.data_backend == 'DALI':
        trainloader = dali.get_dali_loader(trainset, data_freq=opt.datafreq, batch_size=opt.sampled_frames, size=input_dim, 
                                        device_id=opt.gpu_id, num_workers=opt.workers)
    else:
        raise TypeError('unkown data backend {}'.format(opt.data_backend))

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                 collate_fn=test_multibatch_collate_fn)
    # Model
    log.info("creating model")

    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    net.eval()
    if use_gpu:
        net = net.cuda()
        
    # set training parameters
    for p in net.parameters():
        p.requires_grad = True

    criterion = None
    celoss = cross_entropy_loss
    mseloss = torch.nn.SmoothL1Loss(reduction='none')
    
    if opt.loss == 'ce':
        criterion = celoss
    elif opt.loss == 'iou':
        criterion = mask_iou_loss
    elif opt.loss == 'both':
        criterion = [celoss,mask_iou_loss,ref_focal_loss,ref_mse_loss,ref_cos_loss]

    else:
        raise TypeError('unknown training loss %s' % opt.loss)
    criterion_test = pot_criterion
    
    optimizer = None
    
    if opt.solver == 'sgd':

        optimizer = optim.SGD(net.parameters(), lr=opt.learning_rate,
                        momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':
        ignored_params1 = list(map(id, net.Decoder.corner2.parameters()))
        ignored_params2 = list(map(id, net.Decoder.cor_tower.parameters()))
        ignored_params3 = list(map(id, net.Decoder.cls_tower.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params1 and id(p) not in ignored_params2 and id(p) not in ignored_params3, net.parameters())

        optimizer = optim.Adam([{'params': base_params},
                                {'params': net.Decoder.corner2.parameters(), 'lr':opt.learning_rate},
                                {'params': net.Decoder.cor_tower.parameters(), 'lr':opt.learning_rate},
                                {'params': net.Decoder.cls_tower.parameters(), 'lr':opt.learning_rate}],
                                lr=opt.learning_rate,
                                betas=opt.momentum, weight_decay=opt.weight_decay)

    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    # Resume
    title = 'STAN'
    minloss = float('inf')
    max_time = 0.0

    if opt.resume:
        # Load checkpoint.
        log.info('Resuming from checkpoint {}'.format(opt.resume))
        checkpoint = torch.load(opt.resume, map_location='cpu')
        
        try:
            start_epoch = checkpoint['epoch']
            skips = checkpoint['max_skip']
            net.load_param(checkpoint['state_dict'])
        except:
            net.load_param(checkpoint)
        
        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainset.datasets[idx].set_max_skip(skip)
            else:
                # trainloader.dataset.set_max_skip(skip)
                trainset.set_max_skip(skips[0])
        except:
            log.warn('Initializing max skip fail')

    else:
        if opt.initial:
            log.info('Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial, map_location='cpu')
            if isinstance(weight, OrderedDict):
                net.load_param(weight)
            else:
                net.load_param(weight['state_dict'])
        start_epoch = 0
    # apex
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    
    # Train and val
    test_loss = 1e5
    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)
    
    for epoch in range(start_epoch, opt.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        log.info('Skip Info:')
        skip_info = dict()
        if isinstance(trainset, data.ConcatDataset):
            for dataset in trainset.datasets:
                skip_info.update(
                        {type(dataset).__name__: dataset.max_skip}
                    )
        else:
            skip_info.update(
                    {type(trainset).__name__: dataset.max_skip}
                )

        skip_print = ''
        for k, v in skip_info.items():
            skip_print += '{}: {} '.format(k, v)
        log.info(skip_print)
        
        # net.apply(set_bn_eval)
        train_iter, train_loss = train(trainloader,
                           model=net,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=use_gpu,
                           iter_size=opt.iter_size,
                           mode=opt.mode,
                           threshold=opt.iou_threshold,
                           backend=opt.data_backend,
                           print_freq=opt.print_freq,
                           mseloss=mseloss,
                           glob_iter=train_iter)

        # adjust max skip
        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainset, data.ConcatDataset):
                for dataset in trainset.datasets: # trainloader.dataset.datasets:
                    dataset.increase_max_skip()
            else:
                trainset.increase_max_skip()

        # save model
        skips = [ds.max_skip for ds in trainset.datasets]
        
        if test_loss < minloss:
            minloss = test_loss
        if not use_multi:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'max_skip': skips,
            }, epoch + 1, checkpoint=os.path.join(opt.checkpoint, opt.output_dir), filename=opt.mode+'_epoch_'+str(epoch+1))
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.module.state_dict(),
                'max_skip': skips,
            }, epoch + 1, checkpoint=os.path.join(opt.checkpoint, opt.output_dir), filename=opt.mode+'_epoch_'+str(epoch+1))
            
        # save model parameter and gradient
        for name, layer in net.named_parameters():
            if layer.requires_grad == True and layer.grad is not None:               
                # print(name)         
                
                writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), train_iter)
                writer.add_histogram(name + '_data', layer.cpu().data.numpy(), train_iter)
        
        if (epoch + 1) % opt.epoch_per_test == 0:
            test_iter, time_cost, test_loss = test(testloader,
                            model=net,
                            criterion=criterion_test,
                            epoch=epoch,
                            use_cuda=use_gpu,
                            opt=opt,
                            glob_iter=test_iter)

            log.info('results are saved at {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))
        else:
            time_cost = 0.0

        # append logger file
        log_format = 'Epoch: {} LR: {} Train loss: {} Test loss: {} Test Time: {}s'
        log.info(log_format.format(epoch+1, opt.learning_rate, train_loss, test_loss, time_cost))

    log.info('minimum loss: {:f}'.format(minloss))

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, iter_size, mode, threshold, backend, print_freq, mseloss, glob_iter):
    # switch to train mode
    # np_mesh = mesh(1280,720)
    data_time = AverageMeter()
    loss = AverageMeter()
    # coord_rate = 200
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    model.train()
    optimizer.zero_grad()

    cur_loss_ce=None
    cur_loss_iou=None
    cur_loss_coord=None
    cyc_loss_ce=None
    cyc_loss_iou=None
    cyc_loss_coord=None
    pred_corner_max, pred_corner_min = 0, 0
    for batch_idx, data in enumerate(trainloader):
        frames, masks, _, corners, objs, _ = data
        max_obj = masks.shape[2]-1
        data_time.update(time.time() - end)
        end = time.time()

        N, T, C, H, W = frames.size()
        total_loss = 0.0
        

         
        for idx in range(N):
            
            frame = frames[idx]
            mask = masks[idx]
            num_objects = objs[idx]
            corner = corners[idx]
            # print(corner.size())
            no = corner.size(1)
            keys = []
            vals = []

            if glob_iter%10==0 and idx==0:    
                img=[]
            
            for t in range(1, T):
                # memorize
                
                if t-1 == 0:
                    tmp_mask = mask[t-1:t].cuda()
                    r1 = torch.randint(4,(1,1)).item()
                    tmp_corner = corner[t-1].cuda()
                 
                key, val, r4 = model(frame=frame[t-1:t, :, :, :].cuda(), mask=tmp_mask, corner=tmp_corner.roll(r1,1), num_objects=no)
                del tmp_corner, tmp_mask
                torch.cuda.empty_cache()
                 
                # print(key,val,r4)
                keys.append(key)
                vals.append(val)
                # segment
                tmp_key = torch.cat(keys, dim=1)
                tmp_val = torch.cat(vals, dim=1)

                output = model(frame=frame[t:t+1, :, :, :].cuda(), keys=tmp_key, values=tmp_val, num_objects=no, max_obj=max_obj, is_train=True)
                del tmp_key, tmp_val
                torch.cuda.empty_cache()
                
                for k in output.keys():
                    if k[0]=='c':
                        output[k] = output[k].roll(-r1,1)

                
                logits, cur_corner2 = output['p2'], output['c2']
                corner_conf = cur_corner2.max(-1).values.max(-1).values.min().detach().cpu()
                out2 = torch.softmax(logits, dim=1)
                mask_conf = out2[:,1:].max(-1).values.max(-1).values.min().detach().cpu()

                

                # if f:
                cur_loss_ce = criterion[0](out2, mask[t:t+1].cuda(), no, ref=mask[0:1])
                cur_loss_iou = criterion[1](out2, mask[t:t+1].cuda(), no, ref=mask[0:1])
                cur_loss_coord = coord_rate * criterion[3](output['c2'], corner[t].cuda(), no, ref=mask[0:1],mseloss=mseloss)
                total_loss = total_loss + cur_loss_ce + cur_loss_iou + cur_loss_coord

                tmp_mask = out2
                tmp_corner = cur_corner2
                if glob_iter%233==0 and idx==0:
                    img.append(F.interpolate(torch.flip(deNormTensor(frame[t:t+1]),dims=[1])*255,scale_factor=0.5,mode='bilinear',align_corners=False))

                    pred_corner1 = (output['c2'][0:1,0:1]+output['c2'][0:1,1:2]+output['c2'][0:1,2:3]+output['c2'][0:1,3:4])/4+1e-5
                    pred_corner_max, pred_corner_min = pred_corner1.max(), pred_corner1.min()
                    pred_corner1 = F.interpolate(pred_corner1.detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner1.repeat(1,3,1,1)*255)


                    pred_corner1 = F.interpolate(cur_corner2[0].unsqueeze(1).detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner1.repeat(1,3,1,1)*255)                  
                    
                    gt_corner = (corner[t:t+1,0,0:1]+corner[t:t+1,0,1:2]+corner[t:t+1,0,2:3]+corner[t:t+1,0,3:4])/4
                    gt_corner = F.interpolate(gt_corner,scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(gt_corner.repeat(1,3,1,1)*255)
                    
                    pred_corner2 = (output['c2'][1:2,0:1]+output['c2'][1:2,1:2]+output['c2'][1:2,2:3]+output['c2'][1:2,3:4])/4+1e-5
                    pred_corner2 = F.interpolate(pred_corner2.detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner2.repeat(1,3,1,1)*255)
                    if cur_corner2.size(0)>1:
                        pred_corner2 = F.interpolate(cur_corner2[1].unsqueeze(1).detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    else:
                        pred_corner2 = torch.zeros_like(pred_corner1)
                    img.append(pred_corner2.repeat(1,3,1,1)*255)
                    


                    
                    gt_corner1 = (corner[t,1:2,0:1]+corner[t,1:2,1:2]+corner[t,1:2,2:3]+corner[t,1:2,3:4])/4
                    gt_corner1 = F.interpolate(gt_corner1,scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(gt_corner1.repeat(1,3,1,1)*255)
                    del pred_corner1, pred_corner2, gt_corner,gt_corner1
                    torch.cuda.empty_cache()
            
                
                del out2, logits, cur_corner2, output
                torch.cuda.empty_cache()

            # cycle-consistancy
            key, val, r4 = model(frame=frame[T-1:T, :, :, :].cuda(), mask=tmp_mask, corner=tmp_corner.roll(r1,1), num_objects=no)
            keys.append(key)
            vals.append(val)
            del key, val, r4, tmp_mask, tmp_corner
            torch.cuda.empty_cache()
             
            
            cycle_loss = 0.0
            for t in range(T-1, 0, -1): 


                tmp_key = keys[t]
                tmp_val = vals[t]

                output = model(frame=frame[0:1, :, :, :].cuda(), keys=tmp_key, values=tmp_val, num_objects=no, max_obj=max_obj, is_train=True)
                del tmp_key, tmp_val
                torch.cuda.empty_cache()
                for k in output.keys():
                    if k[0]=='c':
                        output[k] = output[k].roll(-r1,1)
                        
                logits, first_corner2 = output['p2'], output['c2']
                first_out2 = torch.softmax(logits, dim=1)

                cyc_loss_ce = criterion[0](first_out2, mask[0:1].cuda(), no, ref=mask[t:t+1]) 
                cyc_loss_iou = criterion[1](first_out2, mask[0:1].cuda(), no, ref=mask[t:t+1]) 
                cyc_loss_coord = coord_rate * criterion[3](output['c2'], corner[0].cuda(), no, ref=mask[0:1],mseloss=mseloss)
                cycle_loss = cycle_loss + cyc_loss_ce + cyc_loss_iou + cyc_loss_coord 
                
                del first_out2, logits
                torch.cuda.empty_cache()
                if glob_iter%233==0 and idx==0:
                    img.append(F.interpolate(torch.flip(deNormTensor(frame[0:1]),dims=[1])*255,scale_factor=0.5,mode='bilinear',align_corners=False))

                    pred_corner1 = (output['c2'][0:1,0:1]+output['c2'][0:1,1:2]+output['c2'][0:1,2:3]+output['c2'][0:1,3:4])/4+1e-5
                    pred_corner_max, pred_corner_min = pred_corner1.max(), pred_corner1.min()
                    pred_corner1 = F.interpolate(pred_corner1.detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner1.repeat(1,3,1,1)*255)


                    pred_corner1 = F.interpolate(first_corner2[0].unsqueeze(1).detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner1.repeat(1,3,1,1)*255)                  
                    
                    gt_corner = (corner[0:1,0,0:1]+corner[0:1,0,1:2]+corner[0:1,0,2:3]+corner[0:1,0,3:4])/4
                    gt_corner = F.interpolate(gt_corner,scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(gt_corner.repeat(1,3,1,1)*255)
                    
                    pred_corner2 = (output['c2'][1:2,0:1]+output['c2'][1:2,1:2]+output['c2'][1:2,2:3]+output['c2'][1:2,3:4])/4+1e-5
                    pred_corner2 = F.interpolate(pred_corner2.detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(pred_corner2.repeat(1,3,1,1)*255)
                    if first_corner2.size(0)>1:
                        pred_corner2 = F.interpolate(first_corner2[1].unsqueeze(1).detach().cpu(),scale_factor=0.5,mode='bilinear',align_corners=False)
                    else:
                        pred_corner2 = torch.zeros_like(pred_corner1)
                    img.append(pred_corner2.repeat(1,3,1,1)*255)
                    


                    
                    gt_corner1 = (corner[0,1:2,0:1]+corner[0,1:2,1:2]+corner[0,1:2,2:3]+corner[0,1:2,3:4])/4
                    gt_corner1 = F.interpolate(gt_corner1,scale_factor=0.5,mode='bilinear',align_corners=False)
                    img.append(gt_corner1.repeat(1,3,1,1)*255)
                    del pred_corner1, pred_corner2, gt_corner,gt_corner1
                    torch.cuda.empty_cache()
                
                del first_corner2, output
                torch.cuda.empty_cache()
            if glob_iter%233==0 and idx==0:
                imgs = torch.cat(img)
                grid_image = make_grid(imgs.int(),nrow=6+7)
                writer.add_image('Train_Iter_Image_corner_gt'+str(coord_rate),grid_image,global_step=glob_iter)
                del img, imgs, grid_image
                torch.cuda.empty_cache()
            total_loss = total_loss + cycle_loss

            del frame, mask, corner, keys, vals
            torch.cuda.empty_cache()
             
        del frames, masks, corners
        torch.cuda.empty_cache()
         
        total_loss = total_loss / (N * (T-1))

        # record loss
        if isinstance(total_loss, torch.Tensor) and total_loss.item() > 0.0:

            # compute gradient and do SGD step (divided by accumulated steps)
            total_loss /= iter_size
            loss.update(total_loss.item(), 1)
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        else:
            print(total_loss.item())
            with amp.scale_loss(cur_loss_ce, optimizer) as scaled_loss:
                scaled_loss.backward()
            model.zero_grad()
            optimizer.zero_grad()
        del total_loss
        torch.cuda.empty_cache()
         
        if (batch_idx+1) % iter_size == 0:
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            glob_iter += 1
        
        if glob_iter % 80 == 0:
    
                writer.add_scalars('tot_loss', {'tot_loss'+str(coord_rate): loss.avg}, glob_iter)
                writer.add_scalars('train_corner_map', {'train_corner_max'+str(coord_rate): pred_corner_max,
                                                        'train_corner_min'+str(coord_rate): pred_corner_min,
                                                        'train_corner_conf'+str(coord_rate): corner_conf,
                                                        'train_mask_conf'+str(coord_rate): mask_conf,}, glob_iter)

                writer.add_scalars('mask_loss', {'cur_ce_loss'+str(coord_rate): cur_loss_ce.item() if not cur_loss_ce is None else 0,
                                                 'cur_iou_loss'+str(coord_rate): cur_loss_iou.item() if not cur_loss_iou is None else 0,
                                                 'cyc_ce_loss'+str(coord_rate): cyc_loss_ce.item() if not cyc_loss_ce is None else 0,
                                                 'cyc_iou_loss'+str(coord_rate): cyc_loss_iou.item() if not cyc_loss_iou is None else 0,}, glob_iter)

                writer.add_scalars('coord_loss', {
                    'cur_loss_coord'+str(coord_rate): cur_loss_coord.item() if not cyc_loss_coord is None else 0,
                    'cyc_loss_coord'+str(coord_rate): cyc_loss_coord.item() if not cyc_loss_coord is None else 0,
                    }, glob_iter)

                
                writer.add_scalars('learning_rate', {'lr'+str(coord_rate): optimizer.state_dict()['param_groups'][0]['lr']}, glob_iter)
        
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f} ce: {ce:.5f} iou: {iou:.5f} coord: {coord:.5f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val[-1],
            loss=loss.avg,
            ce=cur_loss_ce.item() if not cur_loss_ce is None else 0,
            iou=cur_loss_iou.item() if not cur_loss_iou is None else 0,
            coord=cur_loss_coord.item() if not cur_loss_coord is None else 0,
        )
        
        bar.next()
    bar.finish()

    return glob_iter, loss.avg

def test(testloader, model, criterion, epoch, use_cuda, opt, glob_iter):
    data_time = AverageMeter()
    loss = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()
    
    bar = Bar('Processing', max=len(testloader))

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            
            frames, mask, coord, corner, objs, infos = data
            N, T, C, H, W = frames.size()
            coord = coord.view(N,T,-1,4,2)

            frames = frames[0]
            mask = mask[0]
            coord = coord[0]
            corner = corner[0]
            num_objects = objs[0]
            print(num_objects)
            info = infos[0]
            t1 = time.time()
            flag = info['flag']
            

            total_loss = 0.0
            box = np.zeros((T,num_objects,4,2),dtype=np.float32)
            scores = np.zeros((T,num_objects,4),dtype=np.float32)
            firsts = info['frame']['firstframe']

            c=0
            keys = []
            vals = []
            for t in tqdm(range(1, T)):
                if t == 1:
                    tmp_mask = torch.zeros(1,num_objects+1,H,W).cuda()
                    tmp_corner = torch.zeros(num_objects,4,H,W).cuda()
                for ob in range(len(firsts)):
                    if t-1 == firsts[ob]:
                        tmp_mask[0,ob+1] = mask[ob,1].cuda()
                        tmp_mask[0,0] = (tmp_mask[0,ob+1]==0).float() * tmp_mask[0,0]
                        tmp_corner[ob] = corner[ob].cuda()

                if t == 1:
                    tmp_mask[0,0] = (tmp_mask[0,1:].sum(dim=0)==0).float()
                    

                # memorize
                key, val, r4 = model(frame=frames[t-1:t].clone().to(device), mask=tmp_mask, corner=tmp_corner, num_objects=num_objects)

                # segment TODO: deal with sudden num_object change
                tmp_key = torch.cat(keys+[key], dim=1)
                tmp_val = torch.cat(vals+[val], dim=1)
                output = model(frame=frames[t:t+1].clone().to(device), keys=tmp_key, values=tmp_val, num_objects=num_objects, max_obj=num_objects, is_train=True)
                key = key.detach().cpu()
                val = val.detach().cpu()
                logits, corner4 = output['p2'], output['c2']
                del output
                torch.cuda.empty_cache()
                # post processing
                out = torch.softmax(logits, dim=1)
                mask_conf = out[:,1:].max(-1).values.max(-1).values.mean().detach().cpu()
                corner_conf = corner4.max(-1).values.max(-1).values.mean().detach().cpu()
                pred_ord = torch.zeros(num_objects,4,2)
                for ob in range(corner4.size(0)):
                    scores[t,ob] = corner4[ob].detach().cpu().view(4,-1).max(1).values.numpy()
                    for pt in range(4):
                        cm = corner4[ob,pt]
                        idx = torch.argmax(cm)
                        x, y = idx % W, idx / W
                        px = int(math.floor(x + 0.5))
                        py = int(math.floor(y + 0.5))
                        diff=np.zeros(2)
                        if 1 < px < W-1 and 1 < py < H-1: # human prior for peak rather than intepolation
                            diff = np.array([cm[py][px+1] - cm[py][px-1],
                                            cm[py+1][px]-cm[py-1][px]])
                            diff = np.sign(diff)
                        
                        pred_ord[ob,pt,0], pred_ord[ob,pt,1] = x+diff[0], y+diff[1]
                cur_cnts = pred_ord.unsqueeze(0)
                total_loss = total_loss + (criterion(cur_cnts, coord[t:t+1])*flag[t].cpu()).sum()
                c+=flag[t].sum()
                box[t] = cur_cnts.cpu().numpy()*flag[t].view(1,num_objects,1,1).cpu().numpy()
                del cur_cnts
                torch.cuda.empty_cache()

                if (t-1) % opt.save_freq == 0 and corner_conf > 0.5 and mask_conf>0.5:
                    print(len(keys))
                    if len(keys)>26:
                        del keys[-13], vals[-13]
                    keys.append(key.cuda())
                    vals.append(val.cuda())
                del key,val
                torch.cuda.empty_cache()
                
                tmp_mask = out
                tmp_corner = corner4
                
                del corner4,tmp_key,tmp_val,out,r4,logits
                torch.cuda.empty_cache()

            total_loss = total_loss / c
            loss.update(total_loss.item(), 1)            
                
            write_mpot_mask_boxes(box, info, opt, directory=opt.output_dir)
            toc = time.time() - t1
            data_time.update(toc, 1)

            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s |Loss: {loss:.5f}'.format(
                batch=batch_idx+1,
                size=len(testloader),
                data=data_time.val[-1],
                loss=loss.avg
            )
            bar.next()
            
            del box,  coord, frames, mask
            torch.cuda.empty_cache()
            del keys,vals
            torch.cuda.empty_cache()
        bar.finish()
    return glob_iter, data_time.sum, loss.avg

    

if __name__ == '__main__':
    main()
