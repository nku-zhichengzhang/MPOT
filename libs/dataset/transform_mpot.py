import numpy as np
import torch
import math
import cv2

import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from .data import convert_mask, convert_one_hot, MAX_TRAINING_OBJ
from ..utils.utility import pts_resize,Nrotate

class Compose(object):
    """
    Combine several transformation in a serial manner
    """

    def __init__(self, transform=[]):
        self.transforms = transform

    def __call__(self, imgs, annos, coord):

        for m in self.transforms: 
            
            imgs, annos, coord = m(imgs, annos, coord)
            # print(imgs.shape,annos.shape,coord.shape)


        return imgs, annos, coord

class Transpose(object):

    """
    transpose the image and mask
    """

    def __call__(self, imgs, annos, coord):

        H, W, _ = imgs[0].shape
        
        if H <= W:
            return imgs, annos, coord
        else:
            print('transposed!')
            timgs = [np.transpose(img, [1, 0, 2]) for img in imgs]
            tannos = [np.transpose(anno, [0, 2, 1]) for anno in annos]

            return timgs, tannos, coord



class RandomHomo(object):
    
    """
    Affine Transformation to each frame
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, annos, coord):
        # coord: num frame x num objs x 8
        nf,no,_ = coord.shape
        for idx in range(len(imgs)):
            pts = coord[idx,0].copy().reshape(4,2).astype("float32")
            if pts.sum()==0:
                continue
            img = imgs[idx]
            anno = annos[idx]
            
            

            max_obj = anno.shape[2]-1
            h,w,_ = img.shape
            pts_new = np.zeros((4,2),dtype=np.float32)
            off_x, off_y = (2*random.random()-1)*64, (2*random.random()-1)*64

            for i in range(4):
                x = min(20, abs(pts[(i+1)%4,0]-pts[i,0])/64)
                y = min(20, abs(pts[(i+1)%4,1]-pts[i,1])/64)

                # random erasing
                # if random.random()<self.p:
                #     psize = int(min(abs(pts[(i+1)%4,0]-pts[i,0]),abs(pts[(i+1)%4,0]-pts[i,0]))/4)
                #     # print(psize)
                #     ph, pw = int(-random.random()*psize+pts[i,1]), int(-random.random()*psize+pts[i,0])
                #     img[ph:ph+psize,pw:pw+psize] = 0

                # print(x,y)
                pts_new[i,0] = pts[i,0]+(2*random.random()-1)*x + off_x
                pts_new[i,1] = pts[i,1]+(2*random.random()-1)*y + off_y
            
            center_x,center_y = coord[idx,:,0].mean(), coord[idx,:,1].mean()
            angle = 0.5 * (np.random.rand()-0.5) * np.pi
            
            for i in range(4):
                vx,vy = pts_new[i].copy()
                pts_new[i] = Nrotate(angle, vx, vy, center_x, center_y)
            H = cv2.getPerspectiveTransform(pts,pts_new)
            
            coord[idx] = cv2.perspectiveTransform(coord[idx].reshape(-1,1,2),H).reshape(no,8)
            # print(img[ph:ph+psize,pw:pw+psize])
            imgs[idx] = cv2.warpPerspective(img,H,(w,h))

            for i in range(anno.shape[0]):
                m = np.dstack((anno[i,:,:],anno[i,:,:],anno[i,:,:]))
                warp_m = cv2.warpPerspective(np.float32(m),H,(w,h))
                annos[idx,i] = warp_m[:,:,0]
        # print(imgs[0].shape,annos[0].shape)
        return imgs, annos, coord


class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, imgs, annos, coord):
        v = np.random.uniform(-self.delta, self.delta)
        for id, img in enumerate(imgs):
            imgs[id] += v

        return imgs, annos, coord


class randomPeper(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, coord):
        h,w,_ = imgs[0].shape
        noiseNum=int(0.0015*h*w)
        for idx in range(len(imgs)):            
            for i in range(noiseNum):
                randX=random.randint(0,h-1)  
                randY=random.randint(0,w-1)  
                if random.randint(0,1)==0:  
                    imgs[idx][randX,randY]=0  
                else:  
                    imgs[idx][randX,randY]=255 
            
        return imgs, annos, coord


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, imgs, annos, coord):
        v = np.random.uniform(self.lower, self.upper)
        for id, img in enumerate(imgs):
            imgs[id] *= v

        return imgs, annos, coord


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, coord):
        # imgs list of hw3
        # annos nf no h w
        v = random.randint(0, 1)
        if v == 0:
            return imgs, annos, coord

        sample = imgs[0]
        h, w = sample.shape[:2]

        for id, img in enumerate(imgs):
            imgs[id] = img[:, ::-1, :]

        for id, anno in enumerate(annos):
            annos[id] = anno[:, :, ::-1]

        for id, co in enumerate(coord):
            for i in range(4):
                coord[id,:,2*i] = w-co[:,2*i]
        
        return imgs, annos, coord

class ToFloat(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos, coord):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)

        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)
            
        for idx, co in enumerate(coord):
            coord[idx] = co.astype(dtype=np.float32, copy=True)
            
        return imgs, annos, coord

class Rescale(object):

    """
    rescale the size of image and masks
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos, coord):

        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, anno.shape[2]), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno
            annos[id] = canvas

        return imgs, annos, coord

class Stack(object):

    """
    stack adjacent frames into input tensors
    """

    def __call__(self, imgs, annos, coord):

        num_img = len(imgs)
        num_anno = len(annos)

        h, w, = imgs[0].shape[:2]

        # assert num_img == num_anno
        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)

        return img_stack, anno_stack, coord

class ToTensor(object):

    """
    convert to torch.Tensor
    """

    def __call__(self, imgs, annos, coord):
        # imgs nf h w 3
        # annos nf no h w
        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()
        
        # N, H, W, C = imgs.size()
        # for id,co in enumerate(coord):
        #     for i in range(4):
        #         coord[id,2*i] = co[2*i]/W
        #         coord[id,2*i+1] = co[2*i+1]/H
                
        coord = torch.from_numpy(coord.astype(np.float32, copy=True)).float()
        
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.contiguous()
        # print(imgs.size(),annos.size(),coord.size())
        return imgs, annos, coord

class Normalize(object):

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.207, 0.194, 0.199]).reshape([1, 1, 3]).astype(np.float32)

    def __call__(self, imgs, annos, coord):

        for id, img in enumerate(imgs):
            imgs[id] = (img / 255.0 - self.mean) / self.std

        return imgs, annos, coord

class ReverseClip(object):

    def __call__(self, imgs, annos, coord):

        return imgs[::-1], annos[::-1], coord[::-1]

class SampleObject(object):

    def __init__(self, num):
        self.num = num

    def __call__(self, imgs, annos, coord):

        max_obj = annos[0].shape[2] - 1
        num_obj = 0
        while num_obj < max_obj and np.sum(annos[0][:, :, num_obj+1]) > 0:
            num_obj += 1

        if num_obj <= self.num:
            return imgs, annos, coord

        sampled_idx = random.sample(range(1, num_obj+1), self.num)
        sampled_idx.sort()
        for idx, anno in enumerate(annos):
            new_anno = anno.copy()
            new_anno[:, :, self.num+1:] = 0.0
            new_anno[:, :, 1:self.num+1] = anno[:, :, sampled_idx]
            annos[idx] = new_anno

        return imgs, annos, coord

class RandomCrop(object):
    
    def __init__(self, size):
        self.size = size
        
    def __call__(self, imgs, annos, coord):
        
        n,h,w,_ = imgs.shape
        num, ind = 0, 0
        m_coord = np.mean(coord,axis=0)
        center = [np.mean(m_coord[1::2]), np.mean(m_coord[::2])]
        boundary = (50,50)
        
        while num<(3*n) and ind < 10:
            # offset = (random.randint(int(min(max(0, center[0]+50-self.size[0]), h-self.size[0])), int(max(0, min(h-self.size[0], center[0]-50))))\
            #     ,random.randint(int(min(max(0, center[1]+150-self.size[1]), w-self.size[1])), int(max(0, min(w-self.size[1], center[1]-150)))))
            
            offset = [max(0, min(h-self.size[0], random.randint(int(center[0]-boundary[0]-0.5*self.size[0]), int(center[0]+boundary[0]-0.5*self.size[0])))),\
                 max(0, min(w-self.size[1], random.randint(int(center[1]-boundary[1]-0.5*self.size[1]), int(center[1]+boundary[1]-0.5*self.size[1]))))]

            cropped_imgs = imgs[:,offset[0]:offset[0]+self.size[0],offset[1]:offset[1]+self.size[1]]
            cropped_annos = annos[:,offset[0]:offset[0]+self.size[0],offset[1]:offset[1]+self.size[1]]
            cropped_coord = np.zeros(coord.shape)
            num=0
            for id,co in enumerate(coord):
                for i in range(4):
                    cropped_coord[id,2*i] = co[2*i]-offset[1]
                    cropped_coord[id,2*i+1] = co[2*i+1]-offset[0]
                    if cropped_coord[id,2*i]<=w-1 and cropped_coord[id,2*i]>=0 and cropped_coord[id,2*i+1]<=h-1 and cropped_coord[id,2*i+1]>=0:
                        num+=1
            ind+=1


        return cropped_imgs, cropped_annos, cropped_coord


class TrainTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            Transpose(),
            # SampleObject(num=MAX_TRAINING_OBJ),
            RandomHomo(p=0.0),
            ToFloat(),
            RandomContrast(),
            randomPeper(),
            # AdditiveNoise(),
            RandomMirror(),
            # Rescale(size),
            Normalize(),
            Stack(),
            # RandomCrop(size),
            ToTensor(),
        ])

    def __call__(self, imgs, annos, coord):
        return self.transform(imgs, annos, coord)


class TestTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            Transpose(),
            ToFloat(),
            # Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos, coord):
        return self.transform(imgs, annos, coord)

