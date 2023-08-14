from libs.dataset.data import (
        ROOT, 
        build_dataset, 
        test_multibatch_collate_fn, 
    )

from libs.dataset.transform_mpot import TestTransform
from libs.utils.logger import AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, write_mpot_mask_boxes
from libs.models.models import STAN

import torch
import torch.optim as optim
import torch.utils.data as data
import libs.utils.logger as logger

import numpy as np
import os
import time
import cv2

from tensorboardX import SummaryWriter
from progress.bar import Bar
from collections import OrderedDict
from tqdm import tqdm
# from options import OPTION as opt

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
MAX_FLT = 1e6

# parse args
opt, _ = parse_args()
code_id = '0'
tag='res_original_bicubic_corner_aug_'+opt.cfg.split('.')[0]
# Use CUDA
device = 'cuda:{}'.format(code_id)
coord_rate = int(opt.cfg.split('.')[0].split('_')[-1])
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0
split='test'
LOG_DIR = os.path.join(opt.checkpoint,'log')
writer = SummaryWriter(log_dir=LOG_DIR+code_id)
saveroot = '/home/ubuntu/zzc/mpot_lambda_exp/STM-shape/res_original_bicubic_corner_aug_coord_'+str(coord_rate)+'/'+split+'/first/'
if not os.path.exists(saveroot):
    os.makedirs(saveroot)
logger.setup(filename='test_out.log', resume=False)
log = logger.getLogger(__name__)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = code_id

def main():

    # Data
    log.info('Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)
    testset = build_dataset(
        name=opt.valset,
        train=False, 
        split=split,
        tag=tag,
        transform=test_transformer, 
        samples_per_video=1,
        size=input_dim
        )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1,
                                 collate_fn=test_multibatch_collate_fn)
    # Model
    log.info("Creating model")

    net = STAN(opt)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    # set eval to freeze batchnorm update
    net.eval()

    if use_gpu:
        net.to(device)

    # set training parameters
    for p in net.parameters():
        p.requires_grad = False

    # Resume
    title = 'STAN'

    if opt.resume:
        # Load checkpoint.
        log.info('Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        # opt.checkpoint = os.path.dirname(opt.resume)
        checkpoint = torch.load(opt.resume, map_location='cpu')
        
        try:
            net.load_param(checkpoint['state_dict'])
        except:
            net.load_param(checkpoint)
        


    else:
        if opt.initial:
            log.info('Initialize model with weight file {}'.format(opt.initial))
            weight = torch.load(opt.initial, map_location='cpu')
            if isinstance(weight, OrderedDict):
                net.load_param(weight)
            else:
                net.load_param(weight['state_dict'])
    optimizer = optim.Adam(net.parameters(), lr=opt.learning_rate,
                        betas=opt.momentum, weight_decay=opt.weight_decay)
    
    # net, _ = amp.initialize(net, optimizer, opt_level="O1")
    # Train and val
    log.info('Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))

    test(testloader,
        model=net,
        use_cuda=use_gpu,
        opt=opt)

    log.info('Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))

def test(testloader, model, use_cuda, opt):
    criterion = pot_criterion
    data_time = AverageMeter()
    loss = AverageMeter()
    
    bar = Bar('Processing', max=len(testloader))
    pred_corner_max, pred_corner_min = 0, 0
    # model.eval()
    # ignore_dir = [x.split('.')[0] for x in os.listdir(saveroot)]
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            
            frames, mask, _, corner, objs, infos = data
            # if infos[0]['name'] in ignore_dir:
            #     continue    
            N, T, C, H, W = frames.size()
            # coord = coord.view(N,T,-1,4,2).cuda()

            frames = frames[0]
            mask = mask[0]
            # coord = coord[0]
            corner = corner[0]
            num_objects = objs[0]
            # print(num_objects)
            info = infos[0]
            # compute output
            t1 = time.time()
            flag = info['flag']
            scales = []
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
                # for o in range(num_objects):
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
                        # if 1 < px < W-1 and 1 < py < H-1: # human prior for peak rather than intepolation
                        #     diff = np.array([cm[py][px+1] - cm[py][px-1],
                        #                     cm[py+1][px]-cm[py-1][px]])
                        #     diff = np.sign(diff)
                        
                        pred_ord[ob,pt,0], pred_ord[ob,pt,1] = x+diff[0], y+diff[1]
                cur_cnts = pred_ord.cuda().unsqueeze(0)
                # print(coord.shape)
                # total_loss = total_loss + (criterion(cur_cnts, coord[t:t+1])*flag[t].cuda()).sum()
                c+=flag[t].sum()
                box[t] = cur_cnts.cpu().numpy()*flag[t].view(1,num_objects,1,1).cpu().numpy()
                del cur_cnts
                torch.cuda.empty_cache()

                if (t-1) % opt.save_freq == 0 and corner_conf > 0.5 and mask_conf>0.5:
                    # print(len(keys))
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
            
            idx=1
            
            with open(saveroot+info['name']+'.txt','w+')as txt:
                for o in range(num_objects):
                    objects = box[:,o]
                    start = firsts[o]
                    ter=30
                    for t in range(start+1,T):
                        frame = str(int(t+1))
                        c = str(0)    
                        if ter==0:
                            idx+=1
                            ter=30
                        id = str(int(idx))        
                                      
                        if (scores[t,o]>0.3).sum()<3:
                            ter-=1
                        else:
                            ter=30
                            coord = objects[t].copy().reshape(8)
                            coord_txt = ''
                            for i in range(8):
                                coord_txt+="%.2f," % coord[i]
                            info_txt = frame+','+id+','+coord_txt +c+'\n'
                            # info += c+'\n'
                            txt.write(info_txt)
                    idx+=1
                
            # pred = torch.cat(pred, dim=0)
            # pred = pred.detach().cpu().numpy()
            box = np.array(box)
            # print(box.shape)
            write_mpot_mask_boxes(box, info, opt, directory=opt.output_dir)
            # write_mask(pred, info, opt, directory=opt.output_dir)
            toc = time.time() - t1
            log.info(info['name'],toc)

            data_time.update(toc, 1)

            # plot progress
            bar.suffix  = '({batch}/{size}) Time: {data:.3f}s |Loss: {loss:.5f}'.format(
                batch=batch_idx+1,
                size=len(testloader),
                data=data_time.val[-1],
                loss=loss.avg
            )
            bar.next()
            
            del box, frames, mask
            torch.cuda.empty_cache()
            del keys,vals
            torch.cuda.empty_cache()
            
        bar.finish()

    return  data_time.sum, loss.avg


if __name__ == '__main__':
    main()
