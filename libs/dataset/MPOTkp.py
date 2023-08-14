import torch
import os
import os.path as osp
import math
import cv2
import numpy as np

import json
import yaml
import random
import lmdb
import pickle

from PIL import Image
from ..utils.logger import getLogger
from .data import *
from ..utils.utility import build_mpot_mask, build_train_mpot_coord, build_mpot_corner, build_test_mpot_coord, build_test_mpot_mask, pts_resize, gen_offset,deNormTensor
# from ..utils.guass_map import G_map, mesh
from ..utils.guass_torch import make_gaussian, make_mesh

PALETTE = [0, 0, 0, 255, 255, 255, 235, 235, 235, 244, 244, 244, 252, 252,
            252, 247, 247, 247, 204, 204, 204, 208, 208, 208, 193, 193, 193,
            221, 221, 221, 169, 169, 169, 232, 232, 232, 215, 215, 215, 195, 
            195, 195, 111, 111, 111, 143, 143, 143, 79, 79, 79, 162, 162, 162,
            225, 225, 225, 85, 85, 85, 184, 184, 184, 62, 62, 62, 130, 130,
            130, 165, 165, 165, 149, 149, 149, 73, 73, 73, 155, 155, 155, 67,
            67, 67, 178, 178, 178, 90, 90, 90, 101, 101, 101, 39, 39, 39, 135,
            135, 135, 48, 48, 48, 123, 123, 123, 41, 41, 41, 50, 50, 50, 26, 26,
            26, 117, 117, 117, 21, 21, 21, 104, 104, 104, 12, 12, 12, 24, 24, 24,
            43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48,
            49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54,
            55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60,
            61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66,
            67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72,
            73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78,
            79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84,
            85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90,
            91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 
            97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102,
            102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106,
            107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 
            111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116,
            116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 
            121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 
            125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 
            130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 
            135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 
            139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 
            144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 
            149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 
            153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 
            158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 
            163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 
            167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 
            172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176,
            177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181,
            181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186,
            186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 
            191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195,
            195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 
            200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204,
            205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209,
            209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214,
            214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 
            219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 
            223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228,
            228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 
            233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237,
            237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 
            242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 
            247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 
            251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]


FrameRange = [0,1,2,3,4]

class MPOTkpVOS(BaseData):

    def __init__(self, train=True, test_mode='first', split='test', tag=None, sampled_frames=3, transform=None, max_skip=3, increment=4, samples_per_video=150, size=(688,688),max_obj=6):
        data_dir = os.path.join(ROOT, 'MPOT')
        split = 'train' if train else split
        fullfolder = split
        blacklist = dict()
        
        assert test_mode in ['first','biggest']
        self.test_mode = 0 if test_mode == 'first' else 1
        self.root = data_dir
        self.dir = osp.join(data_dir, split)
        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.videos = os.listdir(self.dir)

        self.length = len(self.videos) * samples_per_video
        self.max_obj = max_obj
        self.size = size # h, w
        self.mesh = make_mesh(size[0],size[1])
        self.offset_map = gen_offset(size[1],size[0])
        # self.test_mesh = mesh(1280,720)
        self.transform = transform
        self.train = train
        self.max_skip = max_skip
        self.increment = increment

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]

        imgfolder = os.path.join(self.dir, vid, 'seq1')
        if self.train:
            annotxt = os.path.join(self.dir, vid, 'gt/gt_obj.txt')
            annoinit = np.loadtxt(os.path.join(self.dir, vid, 'gt/gt_obj_init.txt'),dtype=str)
            max_obj = min(self.max_obj,annoinit.shape[0]//2)
        else:
            annotxt = os.path.join(self.dir, vid, 'gt/gt_obj.txt')
            annoinit = np.loadtxt(os.path.join(self.dir, vid, 'gt/gt_obj_init.txt'),dtype=str)
            max_obj = annoinit.shape[0]//2
            # sample_mask = [name.split(',')[0].zfill(6) for name in annoinit[self.test_mode::2]]
            # sample_mask.sort()

        # frames = [name[:5] for name in os.listdir(annofolder) if name not in self.blacklist[vid]]
        frames = [name[:6] for name in os.listdir(imgfolder)]
        frames.sort()
        # del frames[-1]
        nframes = len(frames)
        num_obj = 0
        while num_obj == 0:
            try:
                if self.train:
                    last_sample = -1
                    sample_frame = []
                    firstframe=None
                    sampleframe=None
                    nsamples = min(self.sampled_frames, nframes)
                    obj_list=[]
                    offset = random.choices(FrameRange,weights=[0.5,0.2,0.05,0.05,0.2],k=3)
                    for i in range(nsamples):
                        # if i == 0:
                        #     last_sample = min(int(random.sample(range(0, nframes-nsamples-15), 1)[0]/5)*5+offset[i], nframes-nsamples-15)
                        # else:
                        #     last_sample = min(int(random.sample(
                        #         range(last_sample+1, min(last_sample+self.max_skip, nframes+2*(-nsamples+i))), 1)[0]/5)*5+offset[i], nframes+2*(-nsamples+i))
                        if i == 0:
                                last_sample = random.sample(range(0, nframes-nsamples+1), 1)[0]
                        else:
                            last_sample = random.sample(
                                range(last_sample+1, min(last_sample+self.max_skip+1, nframes-nsamples+i+1)), 1)[0]
                        sample_frame.append(frames[last_sample])

                    frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
                    coord, flag = build_train_mpot_coord([int(name) for name in sample_frame],annotxt,max_obj)
                    if coord is None:
                        continue
                    mask = build_mpot_mask(coord)
                    
                    num_obj = (flag.sum(0)>0).sum()
                    sample_mask = sample_frame
                else:
                    sample_frame = frames
                    sample_mask = [0]
                    nframes = len(sample_frame)

                    frame = [np.array(Image.open(os.path.join(imgfolder, name+'.jpg'))) for name in sample_frame]
                    
                    coord, flag, obj_init = build_test_mpot_coord([int(name) for name in sample_frame],annotxt,annoinit,self.test_mode) 

                    mask, sampleframe, firstframe = build_test_mpot_mask(annoinit,self.test_mode)
                    num_obj = mask.shape[0]
                    # num_obj = max([int(msk.max()) for msk in mask])

                # clear dirty data
                for msk in mask:
                    msk[msk==255] = 0

            except OSError as ose:
                print(ose)
                num_obj = 0
                continue

        if self.train:
            num_obj = min(num_obj, MAX_TRAINING_OBJ)

        info = {'name': vid}
        info['split'] = self.dir
        info['frame'] = {
            'imgs': sample_frame,
            'masks': sample_mask,
            'sampleframe':sampleframe,
            'firstframe':firstframe,
        }

        # if not self.train:
            # assert len(info['frame']['masks']) == len(mask), 'unmatched info-mask pair: {:d} vs {:d} at video {}'.format(len(info['frame']), len(mask), vid)

            # num_ref_mask = len(mask)
            # mask = [mask[0]]
        if not self.train:
            info['obj_ids'] = obj_init
        info['frame']['imgs'].sort()
        info['frame']['masks'].sort()
        info['flag']=torch.from_numpy(flag).int()
        info['palette'] = PALETTE
        info['size'] = frame[0].shape[:2]
        # mask = [convert_mask(msk, self.max_obj) for msk in mask]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')
        # if not self.train:
        frame, mask, coord = self.transform(frame, mask, coord)
            # im = deNormTensor(frame[0])
        
        if self.train:
            t, no, h, w = mask.size()
            no -= 1
            corner = torch.zeros(t, no, 4, h, w)
            for time in range(t):
                for obj in range(no):
                    if flag[time,obj]:
                        corner[time,obj] = build_mpot_corner(coord[time,obj],self.size[::-1],self.mesh)

            return frame, mask, coord, corner, num_obj, info
        else:
            no, _, h, w = mask.size()
            corner = torch.zeros(no, 4, h, w)
            for obj in range(no):
                corner[obj] = build_mpot_corner(coord[sampleframe[obj],obj],self.size[::-1],self.mesh)
            
            return frame, mask, coord, corner, no, info
        
    
    def __len__(self):
        
        return self.length

register_data('MPOTkp', MPOTkpVOS)
# register_data('DALIPOT', DaliPOTVOS)